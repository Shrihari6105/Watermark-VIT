"""WaterMark-VIT command line.

  python cli.py create-exam "CAT1 CSE3501" roster.txt
  python cli.py template out/blank.png [--kind booklet|question] [--seed 3]
  python cli.py issue 0 out/blank.png out/issued/ [--pdf]
  python cli.py investigate leaked_photo.jpg
  python cli.py seal evaluated_scan.png out/sealed.png
  python cli.py verify out/sealed.png [--overlay out/heatmap.png]
  python cli.py attack out/issued/23BCE1001.png phone30deg40cm out/leak.jpg
  python cli.py demo
"""
import argparse
import json
import sys
from pathlib import Path

from watermark_vit import WaterMarkVIT, attacks, imaging, templates

ALL_ATTACKS = {**attacks.DIGITAL, **attacks.PHYSICAL, **attacks.TAMPER}


def _print_investigation(r):
    print(f"Mode       : {r['mode']} (markers found: {r['sync']['markers']})")
    print(f"Pilot      : {r['pilot']}  -> watermark {'present' if r['watermark_present'] else 'NOT detected'}")
    if r["decoded"]:
        d = r["decoded"]
        print(f"Decoded ID : {d['reg_no']}  (exam {d['exam_name']}, roster #{d['roster_idx']}, "
              f"{d['method']}, z = {d['z']} >= {d['z_min']})")
    tr = r.get("tardos")
    if tr:
        print(f"Tardos     : threshold Z = {tr['threshold']} over {tr['effective_len']} positions")
        for s in tr["suspects"]:
            flag = "ACCUSED" if s["accused"] else ""
            print(f"   {s['reg_no']:<14} score {s['score']:>7}  {flag}")
    print(f"VERDICT    : {r['verdict']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="data directory (key + registry)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("create-exam"); p.add_argument("name"); p.add_argument("roster")
    p = sub.add_parser("template"); p.add_argument("out"); p.add_argument("--kind", default="booklet")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("issue"); p.add_argument("exam", type=int); p.add_argument("page")
    p.add_argument("outdir"); p.add_argument("--pdf", action="store_true")
    p = sub.add_parser("investigate"); p.add_argument("image"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("seal"); p.add_argument("scan"); p.add_argument("out")
    p = sub.add_parser("verify"); p.add_argument("sealed"); p.add_argument("--script-id")
    p.add_argument("--overlay")
    p = sub.add_parser("attack"); p.add_argument("image"); p.add_argument("name", choices=sorted(ALL_ATTACKS))
    p.add_argument("out"); p.add_argument("--seed", type=int, default=0)
    sub.add_parser("list-exams")
    sub.add_parser("demo")
    a = ap.parse_args(argv)

    if a.cmd == "template":
        imaging.save(a.out, templates.answer_sheet(a.seed, a.kind))
        print("wrote", a.out)
        return
    if a.cmd == "attack":
        img = imaging.load_gray(a.image)
        imaging.save(a.out, ALL_ATTACKS[a.name](img, seed=a.seed))
        print("wrote", a.out)
        return

    w = WaterMarkVIT(a.data)
    if a.cmd == "create-exam":
        roster = Path(a.roster).read_text().splitlines()
        code = w.reg.create_exam(w.master, a.name, roster)
        print(f"exam code {code} created with {len(w.reg.students(code))} students")
    elif a.cmd == "list-exams":
        for e in w.reg.exams():
            print(f"{e['code']:>3}  {e['name']:<30} n={e['n_students']}  created {e['created']}")
    elif a.cmd == "issue":
        out = Path(a.outdir)
        page = imaging.load_gray(a.page)
        for st, wm in w.issue(page, a.exam):
            imaging.save(out / f"{st['reg_no']}.{'pdf' if a.pdf else 'png'}", wm)
        print("issued copies to", out)
    elif a.cmd == "investigate":
        r = w.investigate(imaging.load_gray(a.image))
        if a.json:
            r = {k: v for k, v in r.items() if k != "rectified"}
            print(json.dumps(r, indent=2, default=str))
        else:
            _print_investigation(r)
    elif a.cmd == "seal":
        sealed, sid = w.seal(imaging.load_gray(a.scan))
        out = Path(a.out).with_suffix(".png")
        imaging.save(out, sealed)
        print(f"sealed script {sid} -> {out} (keep it lossless)")
    elif a.cmd == "verify":
        r = w.verify(imaging.load_gray(a.sealed), a.script_id)
        status = "INTACT" if r["intact"] else f"TAMPERED ({r['tampered_blocks']} / {r['total_blocks']} blocks)"
        print(f"script {r['script_id']} ({r.get('reg_no')}): {status}")
        if a.overlay:
            imaging.save(a.overlay, r["overlay"])
            print("heatmap ->", a.overlay)
        sys.exit(0 if r["intact"] else 2)
    elif a.cmd == "demo":
        demo(w)


def demo(w):
    """Walk the whole lifecycle on synthetic data."""
    out = Path("demo_out")
    roster = [f"23BCE{1000 + i}" for i in range(40)]
    code = w.reg.create_exam(w.master, "DEMO CAT1", roster)
    print(f"[1] exam {code} with {len(roster)} students")
    page = templates.answer_sheet(seed=3, kind="booklet")
    issued = {st["reg_no"]: (st, wm) for st, wm in w.issue(page, code)}
    leaker = roster[17]
    st, wm = issued[leaker]
    imaging.save(out / f"issued_{leaker}.png", wm)
    print(f"[2] issued {len(issued)} watermarked copies (sample: {out}/issued_{leaker}.png)")

    leak = attacks.phone_photo(templates.add_handwriting(wm, seed=1), angle=30, distance=40, seed=5)
    imaging.save(out / "leak_phone_photo.jpg", leak)
    print(f"[3] {leaker} writes on the page and a phone photo leaks -> {out}/leak_phone_photo.jpg")
    _print_investigation(w.investigate(leak))

    a, b = issued[roster[3]][1], issued[roster[9]][1]
    forged = attacks.jpeg(attacks.collude_average(a, b), 70)
    print(f"[4] {roster[3]} and {roster[9]} average their copies to hide:")
    _print_investigation(w.investigate(forged))

    scan = attacks.print_scan(templates.add_handwriting(wm, seed=2, marks=[6, 4, 8, 7, 5]), 300, seed=1)
    sealed, sid = w.seal(scan)
    imaging.save(out / "sealed_script.png", sealed)
    print(f"[5] evaluated script scanned and sealed as {sid}")
    print("    verify untouched :", "INTACT" if w.verify(sealed)["intact"] else "TAMPERED")
    forged = attacks.tamper_marks(sealed, col=1, new_mark="9")
    r = w.verify(forged)
    imaging.save(out / "tamper_heatmap.png", r["overlay"])
    print(f"    Q2 mark 4 -> 9   : {r['tampered_blocks']} blocks flagged -> {out}/tamper_heatmap.png")


if __name__ == "__main__":
    main()
