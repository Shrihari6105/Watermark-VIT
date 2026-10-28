"""Real-world capture kit: print pages, photograph/scan them, measure attribution.

  python fieldkit.py prepare kit/ --pages 12          # printable PDF + digital PNGs + protocol
  ... print, write on the marked pages, scan/photograph, name files P03_phone30_pixel7.jpg ...
  python fieldkit.py analyze kit/ captures/           # writes captures_report/report.md + results.csv

Both steps must use the same --data directory (it holds the key and registry).
"""
import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from watermark_vit import WaterMarkVIT, imaging, templates

CHANNELS = {
    "scan150": "Flatbed scan at 150 dpi (grayscale or colour)",
    "scan300": "Flatbed scan at 300 dpi",
    "phone0": "Phone photo straight above the page, ~40 cm, whole page in frame",
    "phone30": "Phone photo tilted ~30 degrees, ~40 cm",
    "phone45": "Phone photo tilted ~45 degrees, ~60 cm",
    "screen": "Open the page's PNG from digital/ full-screen on a laptop and photograph the screen",
    "whatsapp": "Send a phone0 photo to yourself on WhatsApp/Telegram and save the received image",
    "crop": "Photo or screenshot of only part of the page (no corner markers in frame)",
}
NAME_RE = re.compile(r"^(P\d+)[_-]([A-Za-z0-9]+)(?:[_-]([A-Za-z0-9]+))?", re.I)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
A4_DPI = 155   # canonical 1280x1792 px at 155 dpi fits inside A4 and US Letter


def label_page(img, page_id, write_here):
    out = img.copy()
    text = f"KIT {page_id}" + ("  -  PLEASE WRITE ON THIS PAGE BEFORE CAPTURING" if write_here else "")
    cv2.putText(out, text, (140, 1745), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 2, cv2.LINE_AA)
    return out


def prepare(args):
    w = WaterMarkVIT(args.data)
    out = Path(args.kit)
    (out / "digital").mkdir(parents=True, exist_ok=True)
    roster = [f"KIT{i:04d}" for i in range(args.students)]
    code = w.reg.create_exam(w.master, args.name, roster)
    students = w.reg.students(code)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(students), args.pages, replace=False)

    rows, pages = [], []
    for k, si in enumerate(chosen):
        page_id = f"P{k + 1:02d}"
        kind = "booklet" if k % 2 == 0 else "question"
        write_here = kind == "booklet"
        st = students[int(si)]
        master = label_page(templates.answer_sheet(seed=100 + k, kind=kind), page_id, write_here)
        wm = w.watermark_for(w.prepare_cover(master), code, st["roster_idx"])
        imaging.save(out / "digital" / f"{page_id}.png", wm)
        pages.append(Image.fromarray(wm))
        rows.append({"page_id": page_id, "kind": kind, "write_on_it": int(write_here),
                     "exam_code": code, "roster_idx": st["roster_idx"], "reg_no": st["reg_no"]})

    pages[0].save(out / "print_me.pdf", "PDF", resolution=A4_DPI, save_all=True, append_images=pages[1:])
    with open(out / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "kit.json").write_text(json.dumps({"exam_code": code, "name": args.name,
                                              "data": str(Path(args.data).resolve())}, indent=2))
    write_protocol(out, rows)
    print(f"exam {code} '{args.name}': {len(rows)} pages -> {out}/print_me.pdf")
    print(f"protocol: {out}/PROTOCOL.md")


def write_protocol(out, rows):
    L = ["# WaterMark-VIT field capture protocol", "",
         "1. Print `print_me.pdf` on plain A4/Letter paper, single-sided, with **\"Fit to printable area\" / "
         "\"Shrink to fit\"** so the four black corner markers are printed completely (scale does not "
         "matter; the markers correct for it). Check every corner marker is intact. Note the printer model.",
         "2. On pages marked *PLEASE WRITE ON THIS PAGE*, write a few lines of answers in pen and fill "
         "some marks boxes, like a real student and evaluator would.",
         "3. Capture each page through as many channels as you can (table below). One file per capture.",
         "4. **File names:** `<page>_<channel>_<device>.<ext>`, e.g. `P03_phone30_pixel7.jpg`, "
         "`P08_scan300_hpm428.png`. Extra suffixes are fine: `P03_phone30_pixel7_2.jpg`.",
         "5. Convert HEIC photos to JPG (iPhone: Settings -> Camera -> Formats -> Most Compatible).",
         "6. Put all captures in one folder and run `python fieldkit.py analyze <kit> <folder>`.", "",
         "Keep the kit's `--data` directory: it holds the key and registry needed to decode.", "",
         "## Channels", "", "| Channel tag | How to capture |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in CHANNELS.items()]
    L += ["", "Record lighting and distance per device if you can; report them with the results.",
          "", "## Pages", "", "| Page | Type | Write on it | Suggested channels |", "|---|---|---|---|"]
    tags = list(CHANNELS)
    for i, r in enumerate(rows):
        sugg = ", ".join(f"`{tags[(i + j) % len(tags)]}`" for j in range(4))
        L.append(f"| {r['page_id']} | {r['kind']} | {'yes' if r['write_on_it'] else 'no'} | {sugg} |")
    L += ["", "Minimum useful set: every page through `scan300`, `phone0` and `phone30` on at least two "
          "phones. That gives ≥ 36 physical captures for a 12-page kit."]
    (out / "PROTOCOL.md").write_text("\n".join(L) + "\n")


def analyze(args):
    kit = Path(args.kit)
    meta = json.loads((kit / "kit.json").read_text())
    w = WaterMarkVIT(args.data or meta["data"])
    with open(kit / "manifest.csv") as f:
        manifest = {r["page_id"].upper(): r for r in csv.DictReader(f)}
    out = Path(args.out or Path(args.captures).with_name(Path(args.captures).name + "_report"))
    (out / "failures").mkdir(parents=True, exist_ok=True)

    results = []
    files = sorted(p for p in Path(args.captures).rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if not files:
        raise SystemExit(f"no images found in {args.captures}")
    for p in files:
        m = NAME_RE.match(p.stem)
        page_id, channel, device = (m.group(1).upper(), m.group(2).lower(), (m.group(3) or "unknown").lower()) \
            if m else (None, "unlabelled", "unknown")
        exp = manifest.get(page_id) if page_id else None
        row = {"file": p.name, "page_id": page_id or "", "channel": channel, "device": device,
               "written": exp["write_on_it"] if exp else "", "expected": exp["reg_no"] if exp else ""}
        try:
            img = imaging.load_gray(p)
        except Exception as e:
            results.append({**row, "outcome": "unreadable", "detail": str(e)})
            print(f"{p.name:40s} unreadable ({e})")
            continue
        t = time.time()
        r = w.investigate(img, log=False)
        dec = r["decoded"]
        tr = r.get("tardos") or {}
        rank = ""
        if exp and tr:
            names = [s["reg_no"] for s in tr["suspects"]]
            rank = names.index(exp["reg_no"]) + 1 if exp["reg_no"] in names else ">5"
        got = dec["reg_no"] if dec else ""
        if not exp:
            outcome = "decoded" if dec else "no-decode"
        elif not dec:
            outcome = "missed"
        else:
            outcome = "correct" if got == exp["reg_no"] else "WRONG"
        row.update({"outcome": outcome, "decoded": got, "mode": r["mode"],
                    "markers": len(r["sync"]["markers"]), "pilot": r["pilot"],
                    "z": dec["z"] if dec else "", "tardos_rank": rank,
                    "tardos_accused": int(any(s["accused"] and exp and s["reg_no"] == exp["reg_no"]
                                              for s in tr.get("suspects", []))),
                    "seconds": round(time.time() - t, 2), "verdict": r["verdict"]})
        results.append(row)
        if outcome in ("missed", "WRONG"):
            if r.get("rectified") is not None:
                imaging.save(out / "failures" / f"{p.stem}_rectified.jpg", r["rectified"])
        print(f"{p.name:40s} {outcome:10s} pilot={r['pilot']:<6} {r['verdict']}")

    keys = list(dict.fromkeys(k for r in results for k in r))
    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(results)
    write_report(results, out / "report.md", meta)
    print(f"\nreport -> {out}/report.md")


def _summary(rows):
    lab = [r for r in rows if r.get("expected") and r["outcome"] != "unreadable"]
    n = len(lab)
    ok = sum(r["outcome"] == "correct" for r in lab)
    wrong = sum(r["outcome"] == "WRONG" for r in lab)
    fp = [r for r in lab if r.get("tardos_rank") != ""]
    acc = sum(r["tardos_accused"] for r in fp)
    pct = lambda a, b: f"{100 * a / b:.1f}% ({a}/{b})" if b else "–"
    return n, pct(ok, n), wrong, pct(acc, len(fp))


def write_report(results, path, meta):
    n, acc, wrong, tard = _summary(results)
    L = ["# WaterMark-VIT field test report", "",
         f"Exam: {meta['name']} (code {meta['exam_code']}) · captures analysed: {len(results)}", "",
         "## Overall", "", f"- Labelled captures: **{n}**", f"- Correct student ID: **{acc}**",
         f"- Wrong student named: **{wrong}**", f"- Tardos fingerprint accuses the right student: **{tard}**", ""]
    for title, key in (("By channel", "channel"), ("By device", "device"), ("Written on vs clean", "written")):
        groups = defaultdict(list)
        for r in results:
            groups[str(r.get(key, ""))].append(r)
        L += [f"## {title}", "", f"| {key} | captures | ID correct | wrong ID | Tardos accuses leaker | mean pilot |",
              "|---|---|---|---|---|---|"]
        for g, rows in sorted(groups.items()):
            gn, gacc, gw, gt = _summary(rows)
            pil = [float(r["pilot"]) for r in rows if r.get("pilot") != "" and "pilot" in r]
            label = {"1": "written", "0": "clean", "": "unlabelled"}.get(g, g) if key == "written" else g
            L.append(f"| {label} | {len(rows)} | {gacc} | {gw} | {gt} | {np.mean(pil):.2f} |" if pil else
                     f"| {label} | {len(rows)} | {gacc} | {gw} | {gt} | – |")
        L.append("")
    bad = [r for r in results if r["outcome"] in ("missed", "WRONG", "unreadable")]
    L += ["## Failures", ""]
    if bad:
        L += ["| File | Outcome | Mode | Markers | Pilot | Verdict |", "|---|---|---|---|---|---|"]
        L += [f"| {r['file']} | {r['outcome']} | {r.get('mode', '')} | {r.get('markers', '')} | "
              f"{r.get('pilot', '')} | {r.get('verdict', r.get('detail', ''))} |" for r in bad]
        L += ["", "Rectified views of failed captures are in `failures/`. Pilot < 0.25 (or < 0.45 in blind "
              "mode) means the watermark itself was not found; a high pilot with no ID means the ID bits "
              "were too damaged."]
    else:
        L.append("None.")
    path.write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("kit")
    p.add_argument("--data", default="data")
    p.add_argument("--pages", type=int, default=12)
    p.add_argument("--students", type=int, default=200, help="class size the kit pretends to have")
    p.add_argument("--name", default="FIELD-KIT")
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("analyze")
    p.add_argument("kit")
    p.add_argument("captures")
    p.add_argument("--data", help="defaults to the directory recorded in kit.json")
    p.add_argument("--out")
    a = ap.parse_args(argv)
    prepare(a) if a.cmd == "prepare" else analyze(a)


if __name__ == "__main__":
    main()
