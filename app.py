"""WaterMark-VIT dashboard (Flask).

    python app.py            -> http://127.0.0.1:5050
"""
import base64
import io
import os
import zipfile

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

from watermark_vit import WaterMarkVIT, attacks, imaging, templates

app = Flask(__name__, static_folder="web")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 30 * 1024 * 1024   # images travel as data-URL form fields
W = WaterMarkVIT(os.environ.get("WMVIT_DATA", "data"))

LEAK_CHANNELS = {**attacks.DIGITAL, **attacks.PHYSICAL}


def data_url(img, fmt=".png", max_side=None):
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(fmt, img, [cv2.IMWRITE_JPEG_QUALITY, 92] if fmt == ".jpg" else [])
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf).decode()}"


def uploaded_image(field="file"):
    if field in request.files and request.files[field].filename:
        return imaging.load_gray(request.files[field].read())
    if request.form.get(f"{field}_data"):
        b64 = request.form[f"{field}_data"].split(",", 1)[-1]
        return imaging.load_gray(base64.b64decode(b64))
    return None


def error(msg, code=400):
    return jsonify({"error": msg}), code


@app.get("/")
def index():
    return send_from_directory("web", "index.html")


@app.get("/api/exams")
def exams():
    return jsonify(W.reg.exams())


@app.get("/api/exams/<int:code>/students")
def students(code):
    return jsonify(W.reg.students(code))


@app.post("/api/exams")
def create_exam():
    name = request.form.get("name", "").strip()
    roster = request.form.get("roster", "").splitlines()
    if not name:
        return error("exam name required")
    try:
        code = W.reg.create_exam(W.master, name, roster)
    except ValueError as e:
        return error(str(e))
    return jsonify({"code": code})


@app.get("/api/template")
def template():
    img = templates.answer_sheet(int(request.args.get("seed", 0)), request.args.get("kind", "booklet"))
    return jsonify({"image": data_url(img, max_side=900), "full": data_url(img)})


@app.post("/api/issue")
def issue():
    code = int(request.form["exam"])
    if not W.reg.exam(code):
        return error("unknown exam")
    page = uploaded_image()
    if page is None:
        page = templates.answer_sheet(int(request.form.get("seed", 0)), request.form.get("kind", "booklet"))
    fmt = request.form.get("format", "png")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for st, wm in W.issue(page, code):
            if fmt == "pdf":
                from PIL import Image
                b = io.BytesIO()
                Image.fromarray(wm).save(b, "PDF", resolution=150.0)
                z.writestr(f"{st['reg_no']}.pdf", b.getvalue())
            else:
                z.writestr(f"{st['reg_no']}.png", imaging.encode_png(wm))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"exam{code}_watermarked_{fmt}.zip")


@app.get("/api/channels")
def channels():
    return jsonify({"leak": list(LEAK_CHANNELS), "collusion": list(attacks.COLLUSION),
                    "tamper": list(attacks.TAMPER)})


@app.post("/api/simulate-leak")
def simulate_leak():
    """Attack lab: issue one student's copy, optionally write on it, push it through a channel."""
    code = int(request.form["exam"])
    seed = int(request.form.get("seed", 0))
    page = uploaded_image()
    if page is None:
        page = templates.answer_sheet(seed, request.form.get("kind", "booklet"))
    cover = W.prepare_cover(page)
    idx = [int(x) for x in request.form.getlist("roster_idx")]
    copies = [W.watermark_for(cover, code, i) for i in idx]
    if request.form.get("handwriting") == "1":
        copies = [templates.add_handwriting(c, seed=seed + k, marks=[7, 5, 8]) for k, c in enumerate(copies)]
    if len(copies) == 2:
        img = attacks.COLLUSION[request.form.get("collusion", "average")](copies[0], copies[1])
    else:
        img = copies[0]
    channel = request.form.get("channel", "none")
    img = LEAK_CHANNELS[channel](img, seed=seed)
    fmt = ".jpg" if channel.startswith(("phone", "screen", "jpeg", "whatsapp")) else ".png"
    return jsonify({"image": data_url(img, fmt)})


@app.post("/api/investigate")
def investigate():
    img = uploaded_image()
    if img is None:
        return error("upload an image")
    r = W.investigate(img)
    rect = r.pop("rectified")
    r["rectified"] = data_url(rect, max_side=900) if rect is not None else None
    r["sync"]["markers"] = [int(m) for m in r["sync"]["markers"]]
    if r.get("blind_hypothesis"):
        r["blind_hypothesis"] = [int(v) for v in r["blind_hypothesis"]]
    return jsonify(r)


@app.post("/api/seal")
def seal():
    img = uploaded_image()
    if img is None:
        return error("upload an evaluated scan")
    try:
        sealed, sid = W.seal(img)
    except ValueError as e:
        return error(str(e))
    st = W.reg.script(sid)
    reg = W.reg.student(st["exam_code"], st["roster_idx"])
    return jsonify({"script_id": sid, "reg_no": reg["reg_no"], "image": data_url(sealed)})


@app.post("/api/tamper")
def tamper():
    img = uploaded_image()
    if img is None:
        return error("upload a sealed script")
    name = request.form.get("attack", "marks_overwrite")
    out = attacks.TAMPER[name](img, col=int(request.form.get("col", 1)),
                               new_mark=request.form.get("mark", "9"))
    return jsonify({"image": data_url(out)})


@app.post("/api/verify")
def verify():
    img = uploaded_image()
    if img is None:
        return error("upload a sealed script")
    try:
        r = W.verify(img, request.form.get("script_id") or None)
    except ValueError as e:
        return error(str(e))
    r.pop("tamper_map")
    r["overlay"] = data_url(r["overlay"], ".jpg", max_side=1100)
    return jsonify(r)


@app.get("/api/audit")
def audit():
    return jsonify(W.reg.audit(100))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5050)), debug=False)
