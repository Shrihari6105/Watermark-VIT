"""Run the WaterMark-VIT evaluation matrix and write results/report.md.

    python evaluate.py --pages 12 --students 200
"""
import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from watermark_vit import WaterMarkVIT, attacks, ecc, imaging, metrics, robust, templates


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()) if len(xs) else float("nan"), 0.0
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / np.sqrt(len(xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--students", type=int, default=200)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out)
    shutil.rmtree(out / "eval_data", ignore_errors=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    w = WaterMarkVIT(out / "eval_data")
    rng = np.random.default_rng(args.seed)
    roster = [f"23BCE{1000 + i:04d}" for i in range(args.students)]
    code = w.reg.create_exam(w.master, "EVAL-CAT1", roster)
    n = args.students
    R = {"config": vars(args)}

    # --------------------------------------------------------- imperceptibility
    fid, covers, issued = defaultdict(list), [], []
    t_embed = []
    for i in range(args.pages):
        kind = "booklet" if i % 2 == 0 else "question"
        cover = w.prepare_cover(templates.answer_sheet(seed=i, kind=kind))
        j = int(rng.integers(n))
        t = time.time()
        wm = w.watermark_for(cover, code, j)
        t_embed.append(time.time() - t)
        fid["psnr"].append(metrics.psnr(cover, wm))
        fid["ssim"].append(metrics.ssim(cover, wm))
        covers.append((kind, cover))
        issued.append((kind, j, wm))
        if i < 2:
            imaging.save(out / "samples" / f"issued_{kind}.png", wm)
    R["fidelity"] = {k: ci95(v) for k, v in fid.items()}
    print("fidelity", R["fidelity"])

    # ------------------------------------------------------------- robustness
    channels = {**attacks.DIGITAL, **attacks.PHYSICAL}
    rob = {}
    innocent_trials, innocent_accused = 0, 0
    t_extract = []
    for name, fn in channels.items():
        stats = defaultdict(list)
        for p, (kind, j, wm) in enumerate(issued):
            variants = [("clean", wm)]
            if kind == "booklet":   # Gap 5: student writes on the page after embedding
                variants.append(("hand", templates.add_handwriting(wm, seed=p, marks=[5, 7, 3])))
            for vname, page in variants:
                leaked = fn(page, seed=1000 + p)
                if p == 0 and vname == "hand" and name in ("phone30deg40cm", "screenshoot", "crop30"):
                    imaging.save(out / "samples" / f"leak_{name}.jpg", leaked)
                t = time.time()
                r = w.investigate(leaked, log=False)
                t_extract.append(time.time() - t)
                dec = r["decoded"]
                ok = bool(dec and dec["roster_idx"] == j)
                stats["id_acc"].append(ok)
                stats[f"id_acc_{vname}"].append(ok)
                stats["wrong_id"].append(bool(dec and dec["roster_idx"] != j))
                if r["sync"]["synced"]:
                    canon = imaging.rectify(leaked)[0]
                    soft = robust.extract(canon, w.master, 0)["id_soft"]
                    stats["raw_ber"].append(metrics.ber(soft, ecc.encode(j, code)))
                tr = r.get("tardos")
                if tr:
                    X, p_ = w.codebook(code)
                    fp = robust.extract(imaging.rectify(leaked)[0], w.master, X.shape[1])["fp_soft"]
                    from watermark_vit import tardos
                    sc, eff = tardos.scores(X, p_, fp)
                    Z = tardos.threshold(n, eff, 1e-3)
                    stats["tardos_caught"].append(bool(sc[j] > Z))
                    innocent = np.delete(sc, j)
                    innocent_trials += len(innocent)
                    innocent_accused += int(np.sum(innocent > Z))
        rob[name] = {k: float(np.mean(v)) for k, v in stats.items()}
        print(f"{name:16s}", {k: round(v, 3) for k, v in rob[name].items()})
    R["robustness"] = rob
    R["false_accusation"] = {"innocent_trials": innocent_trials, "innocent_accused": innocent_accused,
                             "rate": innocent_accused / max(innocent_trials, 1)}

    # Unwatermarked pages must decode to nothing and accuse nobody
    unmarked_fp = 0
    for kind, cover in covers:
        for name in ("none", "jpeg70", "printscan300", "phone30deg40cm", "crop50"):
            r = w.investigate(channels[name](cover, seed=7), log=False)
            accused = [s for s in (r.get("tardos") or {}).get("suspects", []) if s["accused"]]
            unmarked_fp += int(bool(r["decoded"]) or bool(accused))
    R["unmarked"] = {"trials": len(covers) * 5, "false_positives": unmarked_fp}
    print("false accusation", R["false_accusation"], "unmarked", R["unmarked"])

    # -------------------------------------------------------------- collusion
    col = defaultdict(lambda: defaultdict(list))
    from watermark_vit import tardos
    X, p_ = w.codebook(code)
    for t in range(args.pages):
        kind, cover = covers[t]
        a, b = rng.choice(n, 2, replace=False)
        wa, wb = w.watermark_for(cover, code, int(a)), w.watermark_for(cover, code, int(b))
        for cname, cfn in attacks.COLLUSION.items():
            for post in ("digital", "jpeg70", "printscan300"):
                forged = cfn(wa, wb)
                if post == "jpeg70":
                    forged = attacks.jpeg(forged, 70)
                elif post == "printscan300":
                    forged = attacks.print_scan(forged, 300, seed=t)
                canon, sync = imaging.rectify(forged)
                fp = robust.extract(canon, w.master, X.shape[1])["fp_soft"]
                sc, eff = tardos.scores(X, p_, fp)
                Z = tardos.threshold(n, eff, 1e-3)
                accused = set(np.flatnonzero(sc > Z))
                key = f"{cname}+{post}"
                col[key]["caught_one"].append(bool(accused & {int(a), int(b)}))
                col[key]["caught_both"].append({int(a), int(b)} <= accused)
                col[key]["innocent_accused"].append(len(accused - {int(a), int(b)}))
                top = set(np.argsort(-sc)[:2])
                col[key]["top2_are_colluders"].append(top == {int(a), int(b)})
    R["collusion"] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in col.items()}
    for k, v in R["collusion"].items():
        print(f"{k:24s}", {m: round(x, 3) for m, x in v.items()})

    # ---------------------------------------------------------------- tamper
    tam = defaultdict(lambda: defaultdict(list))
    false_alarm = []
    jpeg_flagged = []
    t_seal = []
    booklets = [(j, wm) for kind, j, wm in issued if kind == "booklet"]
    for p, (j, wm) in enumerate(booklets):
        written = templates.add_handwriting(wm, seed=50 + p, marks=list(rng.integers(0, 11, 9)))
        scan = attacks.print_scan(written, 300, seed=p)
        t = time.time()
        sealed, sid = w.seal(scan)
        t_seal.append(time.time() - t)
        # lossless round trip must verify clean
        rt = imaging.load_gray(imaging.encode_png(sealed))
        false_alarm.append(int(w.verify(rt)["tampered_blocks"] > 0))
        jpeg_flagged.append(w.verify(attacks.jpeg(sealed, 95), script_id=sid)["tampered_blocks"] /
                            (sealed.size / 256))
        for tname, tfn in attacks.TAMPER.items():
            forged = tfn(sealed, col=int(rng.integers(0, 9)), new_mark=str(int(rng.integers(0, 10))))
            res = w.verify(forged, script_id=sid)
            truth = metrics.block_truth(sealed, forged)
            P, Rc, F, IoU = metrics.prf(res["tamper_map"], truth)
            for m, v in zip(("precision", "recall", "f1", "iou"), (P, Rc, F, IoU)):
                tam[tname][m].append(v)
            if p == 0:
                imaging.save(out / "samples" / f"tamper_{tname}.png", res["overlay"])
        if p == 0:
            imaging.save(out / "samples" / "sealed_script.png", sealed)
    R["tamper"] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in tam.items()}
    R["fragile_false_alarm_rate_png"] = float(np.mean(false_alarm))
    R["fragile_jpeg95_flagged_fraction"] = float(np.mean(jpeg_flagged))
    R["timing_s"] = {"embed": ci95(t_embed), "extract": ci95(t_extract), "seal": ci95(t_seal)}
    print("tamper", R["tamper"])

    (out / "results.json").write_text(json.dumps(R, indent=2, default=float))
    write_report(R, out / "report.md")
    print("wrote", out / "report.md")


def write_report(R, path):
    pct = lambda x: f"{100 * x:.1f}%"
    L = ["# WaterMark-VIT evaluation report", "",
         f"Pages: {R['config']['pages']} (half answer booklets, half question papers) | "
         f"Class size n = {R['config']['students']} | Tardos c = 2, eps = 1e-3", "",
         "Physical channels are **simulated** (see `watermark_vit/attacks.py`); real "
         "printer/phone captures are still required for the final paper.", "",
         "## Imperceptibility", "", "| Metric | Mean ± 95% CI |", "|---|---|"]
    for k, (m, c) in R["fidelity"].items():
        L.append(f"| {k.upper()} | {m:.3f} ± {c:.3f} |")
    L += ["", "## Robustness (leak attribution)", "",
          "`hand` = handwriting + marks added after embedding (embed → write → leak).", "",
          "| Channel | ID acc. (all) | ID acc. clean | ID acc. handwritten | Wrong ID | Raw BER | Tardos catches leaker |",
          "|---|---|---|---|---|---|---|"]
    for name, s in R["robustness"].items():
        g = lambda k: pct(s[k]) if k in s else "–"
        L.append(f"| {name} | {g('id_acc')} | {g('id_acc_clean')} | {g('id_acc_hand')} | {g('wrong_id')} | "
                 f"{s['raw_ber']:.3f} | {g('tardos_caught')} |" if "raw_ber" in s else
                 f"| {name} | {g('id_acc')} | {g('id_acc_clean')} | {g('id_acc_hand')} | {g('wrong_id')} | – | {g('tardos_caught')} |")
    fa = R["false_accusation"]
    L += ["", "## Attribution safety", "",
          f"- Innocent Tardos trials: **{fa['innocent_trials']}**, innocents accused: **{fa['innocent_accused']}** "
          f"(empirical rate {fa['rate']:.2e}; design bound eps/n per trial)",
          f"- Unwatermarked pages decoded or accused: **{R['unmarked']['false_positives']} / {R['unmarked']['trials']}**",
          "", "## Collusion (c = 2)", "",
          "| Attack | ≥1 colluder accused | Both accused | Top-2 = colluders | Innocents accused / trial |",
          "|---|---|---|---|---|"]
    for k, v in R["collusion"].items():
        L.append(f"| {k} | {pct(v['caught_one'])} | {pct(v['caught_both'])} | {pct(v['top2_are_colluders'])} | "
                 f"{v['innocent_accused']:.2f} |")
    L += ["", "## Tamper localisation (fragile layer, 16×16 blocks)", "",
          "| Attack | Precision | Recall | F1 | IoU |", "|---|---|---|---|---|"]
    for k, v in R["tamper"].items():
        L.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['iou']:.3f} |")
    L += ["", f"- False alarms on untouched sealed scripts (PNG round trip): {pct(R['fragile_false_alarm_rate_png'])}",
          f"- Fraction of blocks flagged after a JPEG Q95 re-save: {pct(R['fragile_jpeg95_flagged_fraction'])} "
          "(expected: the fragile layer rejects lossy copies)", "",
          "## Timing (CPU, seconds per page)", "", "| Step | Mean ± 95% CI |", "|---|---|"]
    for k, (m, c) in R["timing_s"].items():
        L.append(f"| {k} | {m:.3f} ± {c:.3f} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
