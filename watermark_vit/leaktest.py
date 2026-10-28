"""Leak test on pages you already have (scans, photos, PDFs) - no printer needed.

    run(["assignment.pdf"], "test_output")

1. watermark  - every page becomes a different student's copy      -> out/pages/P01.png ...
2. leak       - every copy goes through every leak channel         -> out/leaks/P01_whatsapp_sim.jpg ...
3. analyze    - investigate every leak, compare with the true owner -> out/report.md, out/results.csv

Real captures can be added to the leaks folder by hand (e.g. photograph out/pages/P03.png on a
laptop screen and save it as P03_screen_iphone.jpg), then call analyze() again.
"""
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import attacks, imaging
from .pipeline import WaterMarkVIT

# channel tag -> simulated leak. Tags are alphanumeric so analyze() can read them from file names.
CHANNELS = {
    "jpeg50": lambda im, **k: attacks.jpeg(im, 50), "jpeg30": lambda im, **k: attacks.jpeg(im, 30),
    "whatsapp": attacks.whatsapp, "screenshot": attacks.screenshot,
    "resize50": lambda im, **k: attacks.resize(im, 0.5), "rotate3": lambda im, **k: attacks.rotate(im, 3),
    "stirmark": lambda im, **k: attacks.stirmark(im, 2.0, **k),
    "crop50": lambda im, **k: attacks.crop(im, 0.5, **k), "crop30": lambda im, **k: attacks.crop(im, 0.3, **k),
    "screen": lambda im, **k: attacks.screen_shoot(im, 15, **k),
    "phone0": lambda im, **k: attacks.phone_photo(im, 0, 40, **k),
    "phone30": lambda im, **k: attacks.phone_photo(im, 30, 40, **k),
    "phone45": lambda im, **k: attacks.phone_photo(im, 45, 60, **k),
    "scan150": lambda im, **k: attacks.print_scan(im, 150, **k),
    "scan300": lambda im, **k: attacks.print_scan(im, 300, **k),
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
NAME_RE = re.compile(r"^(P\d+)[_-]([A-Za-z0-9]+)(?:[_-]([A-Za-z0-9]+))?", re.I)


def find_images(inputs):
    files = []
    for src in map(Path, inputs):
        files += sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT) if src.is_dir() else [src]
    return files


def watermark(inputs, out, students=200, seed=7):
    """Give every input page a different (pseudo-)student's watermark."""
    pages = [(f, i, pg) for f in find_images(inputs) for i, pg in enumerate(imaging.load_pages(f))]
    if not pages:
        raise SystemExit("no pages found in the inputs")
    w = WaterMarkVIT(out / "data")
    roster = [f"STU{i:04d}" for i in range(max(students, len(pages)))]
    code = w.reg.create_exam(w.master, "LEAK TEST", roster)
    chosen = np.random.default_rng(seed).choice(len(roster), len(pages), replace=False)
    rows = []
    for k, ((f, i, pg), si) in enumerate(zip(pages, chosen)):
        st = w.reg.students(code)[int(si)]
        page_id = f"P{k + 1:02d}"
        imaging.save(out / "pages" / f"{page_id}.png", w.watermark_for(w.prepare_cover(pg), code, st["roster_idx"]))
        rows.append({"page_id": page_id, "source": f"{f.name} p{i + 1}", "owner": st["reg_no"]})
        print(f"{page_id}  {f.name} page {i + 1}  ->  {st['reg_no']}")
    with open(out / "owners.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def leak(out, channels=None):
    """Push every watermarked page through every simulated leak channel."""
    chans = channels or list(CHANNELS)
    for k, p in enumerate(sorted((out / "pages").glob("P*.png"))):
        img = imaging.load_gray(p)
        for j, c in enumerate(chans):
            lossless = c.startswith(("crop", "resize", "rotate", "scan"))
            imaging.save(out / "leaks" / f"{p.stem}_{c}_sim.{'png' if lossless else 'jpg'}",
                         CHANNELS[c](img, seed=1000 * k + j))
    print(f"leaked {len(chans)} ways -> {out / 'leaks'}")


def analyze(out):
    """Investigate every file in out/leaks and score it against the true owner."""
    w = WaterMarkVIT(out / "data")
    with open(out / "owners.csv") as fh:
        owners = {r["page_id"]: r["owner"] for r in csv.DictReader(fh)}
    results = []
    for p in find_images([out / "leaks"]):
        m = NAME_RE.match(p.stem)
        page_id, channel, device = (m.group(1).upper(), m.group(2).lower(), (m.group(3) or "?").lower()) \
            if m else ("", "unlabelled", "?")
        row = {"file": p.name, "channel": channel, "device": device, "owner": owners.get(page_id, "")}
        try:
            r = w.investigate(imaging.load_gray(p), log=False)
        except Exception as e:
            results.append({**row, "outcome": "unreadable", "verdict": str(e)})
            continue
        got = r["decoded"]["reg_no"] if r["decoded"] else ""
        outcome = ("decoded" if got else "no-decode") if not row["owner"] else \
            "missed" if not got else "correct" if got == row["owner"] else "WRONG"
        results.append({**row, "outcome": outcome, "decoded": got, "pilot": r["pilot"], "verdict": r["verdict"]})
        print(f"{p.name:32s} {outcome:8s} {r['verdict']}")
    with open(out / "results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in results for k in r)))
        wr.writeheader()
        wr.writerows(results)
    write_report(results, out / "report.md")
    return results


def write_report(results, path):
    scored = [r for r in results if r["owner"]]
    ok = sum(r["outcome"] == "correct" for r in scored)
    wrong = sum(r["outcome"] == "WRONG" for r in scored)
    L = ["# Leak test report", "",
         f"**{ok}/{len(scored)} leaked copies traced to the right student. Wrong student named: {wrong}.**", "",
         "| Channel | Copies | Traced correctly | Wrong student | Mean pilot |", "|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in scored:
        groups[r["channel"]].append(r)
    for ch, rows in sorted(groups.items()):
        pil = [r["pilot"] for r in rows if "pilot" in r]
        L.append(f"| {ch} | {len(rows)} | {sum(r['outcome'] == 'correct' for r in rows)} | "
                 f"{sum(r['outcome'] == 'WRONG' for r in rows)} | {np.mean(pil):.2f} |" if pil else
                 f"| {ch} | {len(rows)} | 0 | 0 | - |")
    bad = [r for r in results if r["outcome"] not in ("correct", "decoded")]
    L += ["", "## Not traced", ""]
    L += [f"- `{r['file']}`: {r['outcome']} - {r['verdict']}" for r in bad] or ["None."]
    L += ["", "Pilot = strength of the hidden watermark signal (present above 0.25 with corner markers, "
          "0.45 without). `_sim` files are computer-simulated leaks, not real captures."]
    path.write_text("\n".join(L) + "\n")


def run(inputs, out="test_output", channels=None):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    watermark(inputs, out)
    leak(out, channels)
    results = analyze(out)
    scored = [r for r in results if r["owner"]]
    ok = sum(r["outcome"] == "correct" for r in scored)
    wrong = sum(r["outcome"] == "WRONG" for r in scored)
    print(f"\n{'=' * 60}\nRESULT: {ok}/{len(scored)} leaked copies traced to the right student, "
          f"{wrong} wrong\nReport: {out / 'report.md'}\n{'=' * 60}")
    return results
