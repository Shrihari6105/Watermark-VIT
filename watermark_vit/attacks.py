"""Attack / leak-channel simulation harness.

Digital: JPEG, WebP, resize, crop, rotation, noise, blur, StirMark-style
bending, WhatsApp round trip, 1080p screenshot.
Physical (simulated): print->scan, print->phone photo, screen->phone photo.
Collusion: averaging, splicing, block interleaving.
Tamper: marks overwrite, copy-move, inpainting, erasure.

Physical simulations are approximations for development; the evaluation
plan calls for real printer/phone captures as the authoritative test.
"""
import cv2
import numpy as np

from .templates import marks_cell, write_marks


def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- digital
def jpeg(img, q=75, **_):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])[1], 0)


def webp(img, q=75, **_):
    return cv2.imdecode(cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, int(q)])[1], 0)


def resize(img, s=0.5, **_):
    h, w = img.shape
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=interp)


def crop(img, keep=0.5, seed=0, **_):
    """Keep a random axis-aligned window holding `keep` of the page area."""
    rng = _rng(seed)
    h, w = img.shape
    side = np.sqrt(keep)
    ch, cw = int(h * side), int(w * side)
    y = int(rng.integers(0, h - ch + 1))
    x = int(rng.integers(0, w - cw + 1))
    return img[y:y + ch, x:x + cw].copy()


def rotate(img, deg=3.0, **_):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)


def noise(img, sigma=8.0, seed=0, **_):
    n = _rng(seed).normal(0, sigma, img.shape)
    return np.clip(img + n, 0, 255).astype(np.uint8)


def blur(img, k=5, **_):
    return cv2.GaussianBlur(img, (k, k), 0)


def stirmark(img, amp=2.0, seed=0, **_):
    """Random smooth bending (StirMark-like geometric desynchronisation)."""
    rng = _rng(seed)
    h, w = img.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = rng.normal(0, 1, (6, 6)).astype(np.float32)
    fy = rng.normal(0, 1, (6, 6)).astype(np.float32)
    dx = cv2.resize(fx, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    dy = cv2.resize(fy, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    return cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=255)


def whatsapp(img, **_):
    """Typical messenger pipeline: downscale to <=1600 px long side, JPEG ~Q60."""
    h, w = img.shape
    s = min(1.0, 1600 / max(h, w))
    return jpeg(resize(img, s) if s < 1 else img, 60)


def screenshot(img, **_):
    """Page shown full-height on a 1080p display, then screenshotted."""
    return resize(img, 1000 / img.shape[0])


# --------------------------------------------------------------- physical
def _tone(img, gamma, lo, hi):
    f = (img.astype(np.float32) / 255.0) ** gamma
    return lo + f * (hi - lo)


def print_scan(img, dpi=300, seed=0, **_):
    """Laser print + flatbed scan: dot gain, tone curve, skew, paper noise, resampling."""
    rng = _rng(seed)
    s = dpi / 150.0
    up = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 0.7 * s)                     # toner spread
    f = _tone(up, rng.uniform(0.85, 1.2), rng.uniform(15, 35), rng.uniform(225, 250))
    f += cv2.resize(rng.normal(0, 4, (up.shape[0] // 4, up.shape[1] // 4)).astype(np.float32),
                    (up.shape[1], up.shape[0]))                    # paper grain
    f += rng.normal(0, 2.5, f.shape)                                # scanner noise
    out = np.clip(f, 0, 255).astype(np.uint8)
    # Pad onto the scanner bed and skew slightly
    pad = int(40 * s)
    bed = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=250)
    deg = rng.uniform(-1.5, 1.5)
    h, w = bed.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    M[:, 2] += rng.uniform(-10, 10, 2) * s
    return cv2.warpAffine(bed, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=250)


def _camera(img, angle, distance_scale, seed, light=True):
    rng = _rng(seed)
    h, w = img.shape
    # Page placed on a darker desk
    pad = int(0.12 * max(h, w))
    scene = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=70)
    H, W = scene.shape
    # Perspective: tilt about the horizontal axis by `angle` degrees + jitter
    t = np.tan(np.radians(angle)) * 0.35
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[W * t / 2, 0], [W - W * t / 2, 0], [W, H], [0, H]])
    dst += rng.uniform(-0.015, 0.015, dst.shape).astype(np.float32) * [W, H]
    out_w, out_h = int(W * distance_scale), int(H * distance_scale)
    M = cv2.getPerspectiveTransform(src, (dst * distance_scale).astype(np.float32))
    shot = cv2.warpPerspective(scene, M, (out_w, out_h), flags=cv2.INTER_AREA,
                               borderValue=60).astype(np.float32)
    if light:
        yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        cx, cy = rng.uniform(0.2, 0.8) * out_w, rng.uniform(0.2, 0.8) * out_h
        r = np.sqrt(((xx - cx) / out_w) ** 2 + ((yy - cy) / out_h) ** 2)
        shot *= (1.05 - 0.35 * r)                                   # vignetting / uneven light
        shot = _tone(np.clip(shot, 0, 255), rng.uniform(0.9, 1.15), 10, 240)
    shot = cv2.GaussianBlur(shot, (0, 0), rng.uniform(0.5, 1.0))   # defocus
    shot += rng.normal(0, 3.5, shot.shape)                          # sensor noise
    return jpeg(np.clip(shot, 0, 255).astype(np.uint8), 85)


def phone_photo(img, angle=0, distance=40, seed=0, **_):
    """Print -> phone photo. `distance` (cm) controls effective resolution."""
    printed = print_scan(img, dpi=150, seed=seed)
    scale = np.clip(40.0 / distance, 0.6, 1.4)
    return _camera(printed, angle, scale, seed + 1)


def screen_shoot(img, angle=15, seed=0, **_):
    """Displayed on a laptop screen and photographed: gamma, moire, perspective."""
    rng = _rng(seed)
    disp = resize(img, 1000 / img.shape[0])
    disp = cv2.resize(disp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    f = _tone(disp, 1.1, 25, 235)
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = rng.uniform(0, np.pi)
    moire = np.sin((xx * np.cos(th) + yy * np.sin(th)) * rng.uniform(0.35, 0.6))
    f += 6 * moire
    return _camera(np.clip(f, 0, 255).astype(np.uint8), angle, 1.0, seed + 7)


# --------------------------------------------------------------- collusion
def collude_average(a, b, **_):
    return ((a.astype(np.uint16) + b) // 2).astype(np.uint8)


def collude_splice(a, b, **_):
    out = a.copy()
    out[a.shape[0] // 2:] = b[a.shape[0] // 2:]
    return out


def collude_interleave(a, b, block=64, **_):
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // block) + (xx // block)) % 2 == 0, a, b).astype(np.uint8)


# --------------------------------------------------------------- tampering
def tamper_marks(img, col=0, new_mark="9", **_):
    """Erase one marks cell and write a different number."""
    out = img.copy()
    x0, y0, x1, y1 = marks_cell(col)
    paper = int(np.median(out[y0 + 5:y1 - 5, x0 + 5:x1 - 5][out[y0 + 5:y1 - 5, x0 + 5:x1 - 5] > 128]))
    out[y0 + 4:y1 - 3, x0 + 4:x1 - 3] = paper
    marks = [""] * col + [new_mark]
    write_marks(out, marks, ink=28)
    return out


def tamper_copy_move(img, src=(220, 600, 500, 120), dst=(220, 1100), **_):
    out = img.copy()
    x, y, w, h = src
    dx, dy = dst
    out[dy:dy + h, dx:dx + w] = img[y:y + h, x:x + w]
    return out


def tamper_inpaint(img, region=(300, 800, 400, 90), **_):
    x, y, w, h = region
    mask = np.zeros_like(img)
    mask[y:y + h, x:x + w] = 255
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def tamper_erase(img, region=(600, 1300, 350, 60), **_):
    out = img.copy()
    x, y, w, h = region
    out[y:y + h, x:x + w] = 243
    return out


DIGITAL = {
    "none": lambda im, **k: im,
    "jpeg90": lambda im, **k: jpeg(im, 90), "jpeg70": lambda im, **k: jpeg(im, 70),
    "jpeg50": lambda im, **k: jpeg(im, 50), "jpeg30": lambda im, **k: jpeg(im, 30),
    "webp60": lambda im, **k: webp(im, 60),
    "resize0.5": lambda im, **k: resize(im, 0.5), "resize1.5": lambda im, **k: resize(im, 1.5),
    "rotate3": lambda im, **k: rotate(im, 3), "rotate-5": lambda im, **k: rotate(im, -5),
    "noise8": lambda im, **k: noise(im, 8, **k), "blur5": lambda im, **k: blur(im, 5),
    "stirmark": lambda im, **k: stirmark(im, 2.0, **k),
    "whatsapp": whatsapp, "screenshot1080": screenshot,
    "crop50": lambda im, **k: crop(im, 0.5, **k), "crop30": lambda im, **k: crop(im, 0.3, **k),
}
PHYSICAL = {
    "printscan150": lambda im, **k: print_scan(im, 150, **k),
    "printscan300": lambda im, **k: print_scan(im, 300, **k),
    "phone0deg40cm": lambda im, **k: phone_photo(im, 0, 40, **k),
    "phone30deg40cm": lambda im, **k: phone_photo(im, 30, 40, **k),
    "phone45deg60cm": lambda im, **k: phone_photo(im, 45, 60, **k),
    "screenshoot": lambda im, **k: screen_shoot(im, 15, **k),
}
COLLUSION = {"average": collude_average, "splice": collude_splice, "interleave": collude_interleave}
TAMPER = {"marks_overwrite": tamper_marks, "copy_move": tamper_copy_move,
          "inpaint": tamper_inpaint, "erase": tamper_erase}
