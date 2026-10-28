"""High-level WaterMark-VIT workflow.

Lifecycle split (the core design idea):
  1. issue()       before the exam   - robust per-student ID + Tardos code on paper
  2. investigate() after a leak      - rectify, decode ID, rank Tardos suspects
  3. seal()        after evaluation  - fragile HMAC layer on the scanned script
  4. verify()      on dispute        - block-level tamper map + whose script it is
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import norm

from . import ecc, fragile, imaging, robust, tardos
from .config import PAGE_H, PAGE_W, PILOT_MIN, PILOT_MIN_BLIND, TARDOS_C, TARDOS_EPS
from .keys import load_or_create_key
from .registry import Registry

# ID acceptance test. For a page without this student's mark, the normalised
# correlation z = <c, s> / ||s|| with any fixed bipolar codeword c is ~N(0, 1);
# z must clear a Bonferroni bound over all registered codewords so the
# family-wise probability of naming anyone from noise is <= ID_ALPHA.
ID_ALPHA = 1e-4
ID_MIN_LEAD = 0.75          # z lead over the runner-up (two colluders tie -> no single ID)


def sha256(img):
    return hashlib.sha256(img.tobytes()).hexdigest()


class WaterMarkVIT:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.master = load_or_create_key(self.data_dir / "master.key")
        self.reg = Registry(self.data_dir / "registry.sqlite")
        self._codebooks = {}

    # ------------------------------------------------------------------ setup
    def codebook(self, exam_code):
        if exam_code not in self._codebooks:
            exam = self.reg.exam(exam_code)
            X, p = tardos.codebook(self.master, f"exam-{exam_code}", exam["n_students"],
                                   exam["tardos_len"], TARDOS_C)
            self._codebooks[exam_code] = (X, p)
        return self._codebooks[exam_code]

    @staticmethod
    def prepare_cover(img):
        """Canonical size + paper tint + fiducials (no watermark yet)."""
        if img.shape != (PAGE_H, PAGE_W):
            img = imaging.to_canonical(img)
        return imaging.add_fiducials(imaging.tint(img))

    # ------------------------------------------------------------------ issue
    def watermark_for(self, cover, exam_code, roster_idx):
        X, _ = self.codebook(exam_code)
        id_bits = ecc.encode(roster_idx, exam_code)
        return robust.embed(cover, self.master, id_bits, X[roster_idx])

    def issue(self, master_page, exam_code):
        """Yield (student_row, watermarked page) for every student of the exam."""
        cover = self.prepare_cover(master_page)
        for st in self.reg.students(exam_code):
            page = self.watermark_for(cover, exam_code, st["roster_idx"])
            self.reg.mark_issued(exam_code, st["roster_idx"], sha256(page))
            yield st, page
        self.reg.log("issue", exam=exam_code)

    # ------------------------------------------------------------ investigate
    def _tardos_report(self, exam_code, fp_soft, top=5):
        X, p = self.codebook(exam_code)
        sc, eff = tardos.scores(X, p, fp_soft)
        Z = tardos.threshold(len(X), eff, TARDOS_EPS)
        order = np.argsort(-sc)[:top]
        suspects = []
        for j in order:
            st = self.reg.student(exam_code, int(j)) or {}
            suspects.append({"roster_idx": int(j), "reg_no": st.get("reg_no"),
                             "score": round(float(sc[j]), 1), "accused": bool(sc[j] > Z)})
        return {"threshold": round(Z, 1), "effective_len": eff, "suspects": suspects,
                "eps": TARDOS_EPS}

    def _registered_codewords(self):
        """All issued ID codewords as a bipolar matrix, for ML (list) decoding."""
        exams = self.reg.exams()
        key = tuple((e["code"], e["n_students"]) for e in exams)
        if getattr(self, "_cw_key", None) != key:
            ids, rows = [], []
            for e in exams:
                for j in range(e["n_students"]):
                    ids.append((e["code"], j))
                    rows.append(2.0 * ecc.encode(j, e["code"]) - 1)
            self._cw_key, self._cw = key, (ids, np.array(rows).reshape(-1, ecc.CODE_BITS))
        return self._cw

    def decode_id(self, id_soft):
        """Identify the student from soft ID bits.

        Reed-Solomon proposes a codeword without needing the registry; the
        decision is a correlation test against every registered codeword, so a
        bounded-distance RS "success" on noise cannot name anyone by itself.
        """
        ids, B = self._registered_codewords()
        if not ids:
            return None
        s = np.clip(id_soft, -1, 1)
        z = B @ s / (np.linalg.norm(s) + 1e-12)
        z_min = float(norm.ppf(1 - ID_ALPHA / len(ids)))
        order = np.argsort(-z)
        best = int(order[0])
        runner_up = float(z[order[1]]) if len(order) > 1 else 0.0
        if z[best] < z_min or z[best] - runner_up < ID_MIN_LEAD:
            return None
        code, j = ids[best]
        rs = ecc.decode_soft(id_soft)
        rs_agrees = bool(rs and (rs["exam_code"], rs["roster_idx"]) == (code, j))
        return {"exam_code": code, "roster_idx": j,
                "method": "RS + registry test" if rs_agrees else "registry ML test",
                "z": round(float(z[best]), 2), "z_min": round(z_min, 2),
                "lead": round(float(z[best]) - runner_up, 2),
                "bit_errors": int(np.sum((B[best] > 0) != (s > 0))),
                "erasures": rs["erasures"] if rs_agrees else 0}

    def investigate(self, img, log=True):
        """Attribute a leaked copy (photo, scan, screenshot, crop)."""
        canon, sync = imaging.rectify(img)
        result = {"sync": sync, "mode": "fiducials" if sync["synced"] else "blind",
                  "rectified": canon if sync["synced"] else None}
        if sync["synced"]:
            ext = robust.extract(canon, self.master, 0)
            pilot, id_soft = ext["pilot"], ext["id_soft"]
        else:
            ext = None
            b = robust.blind_extract(canon, self.master)
            pilot, id_soft = b["pilot"], b["id_soft"]
            result["blind_hypothesis"] = b["hypothesis"]
        result["pilot"] = round(float(pilot), 3)
        result["watermark_present"] = bool(pilot >= (PILOT_MIN if sync["synced"] else PILOT_MIN_BLIND))
        result["confidence"] = round(float(np.mean(np.minimum(np.abs(id_soft), 1))), 3)

        dec = self.decode_id(id_soft) if result["watermark_present"] else None
        if dec:
            exam = self.reg.exam(dec["exam_code"])
            st = self.reg.student(dec["exam_code"], dec["roster_idx"])
            result["decoded"] = {"exam_code": dec["exam_code"], "exam_name": exam["name"],
                                 "roster_idx": dec["roster_idx"], "reg_no": st["reg_no"],
                                 "method": dec["method"], "z": dec["z"], "z_min": dec["z_min"],
                                 "bit_errors": dec["bit_errors"], "erasures": dec["erasures"]}
        else:
            result["decoded"] = None

        if ext is not None and result["watermark_present"]:
            exams = [result["decoded"]["exam_code"]] if result["decoded"] else \
                [e["code"] for e in self.reg.exams()]
            best = None
            for code in exams:
                L = self.reg.exam(code)["tardos_len"]
                fp_soft = robust.extract(canon, self.master, L)["fp_soft"]
                rep = self._tardos_report(code, fp_soft)
                rep["exam_code"] = code
                if best is None or rep["suspects"][0]["score"] > best["suspects"][0]["score"]:
                    best = rep
            result["tardos"] = best

        result["verdict"] = self._verdict(result)
        if log:
            self.reg.log("investigate", mode=result["mode"], verdict=result["verdict"],
                         decoded=result["decoded"])
        return result

    @staticmethod
    def _verdict(r):
        dec, tr = r.get("decoded"), r.get("tardos")
        accused = [s for s in (tr or {}).get("suspects", []) if s["accused"]]
        names = ", ".join(s["reg_no"] for s in accused)
        if not r.get("watermark_present", True):
            return "No watermark detected"
        if len(accused) >= 2:
            return f"Collusion detected - fingerprint accuses {names}"
        if dec and (tr is None or any(s["roster_idx"] == dec["roster_idx"] for s in accused)):
            return f"Leak traced to {dec['reg_no']}"
        if dec and accused:
            return f"ID decodes to {dec['reg_no']} but fingerprint accuses {names} - possible collusion"
        if accused:
            return f"Fingerprint accuses {names} (ID layer unreadable - possible collusion)"
        if dec:
            return f"ID decodes to {dec['reg_no']} (fingerprint below accusation threshold)"
        return "No watermark recovered"

    # ------------------------------------------------------------ seal/verify
    @staticmethod
    def script_id(exam_code, roster_idx):
        return f"E{exam_code:02d}-S{roster_idx:04d}"

    def seal(self, scan):
        """Rectify an evaluated scan, identify its owner, add the fragile layer."""
        canon, sync = imaging.rectify(scan)
        if not sync["synced"]:
            canon = imaging.to_canonical(canon)
        ext = robust.extract(canon, self.master, 0)
        dec = self.decode_id(ext["id_soft"]) if ext["pilot"] >= PILOT_MIN else None
        if not dec:
            raise ValueError("could not identify the script owner from the robust watermark")
        sid = self.script_id(dec["exam_code"], dec["roster_idx"])
        sealed = fragile.seal(canon, self.master, sid)
        self.reg.add_script(sid, dec["exam_code"], dec["roster_idx"], sha256(sealed))
        self.reg.log("seal", script_id=sid)
        return sealed, sid

    def verify(self, sealed, script_id=None):
        if sealed.shape != (PAGE_H, PAGE_W):
            raise ValueError("sealed scripts are canonical PNGs; got a different size "
                             "(re-scanned or resized copies cannot be verified)")
        if script_id is None:
            dec = self.decode_id(robust.extract(sealed, self.master, 0)["id_soft"])
            if not dec:
                raise ValueError("could not read script ID; supply it explicitly")
            script_id = self.script_id(dec["exam_code"], dec["roster_idx"])
        tmap = fragile.verify(sealed, self.master, script_id)
        rec = self.reg.script(script_id)
        out = {"script_id": script_id, "registered": rec is not None,
               "tampered_blocks": int(tmap.sum()), "total_blocks": int(tmap.size),
               "intact": bool(tmap.sum() == 0), "tamper_map": tmap,
               "overlay": fragile.heatmap_overlay(sealed, tmap)}
        if rec:
            st = self.reg.student(rec["exam_code"], rec["roster_idx"])
            out["reg_no"] = st["reg_no"] if st else None
        self.reg.log("verify", script_id=script_id, tampered=out["tampered_blocks"])
        return out
