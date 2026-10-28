"""Page I/O, canonicalisation, corner fiducials and geometric re-synchronisation."""
from pathlib import Path

import cv2
import numpy as np

from .config import INK_BLACK, MARKER_MARGIN, MARKER_SIZE, PAGE_H, PAGE_W, PAPER_WHITE

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_DETECTOR = cv2.aruco.ArucoDetector(_DICT, _params)


def load_gray(path_or_bytes):
    """Load an image (or first page of a PDF) as uint8 grayscale."""
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        data = path.read_bytes()
        is_pdf = path.suffix.lower() == ".pdf"
    else:
        data = path_or_bytes
        is_pdf = data[:4] == b"%PDF"
    if is_pdf:
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not decode image")
    return img


def save(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        from PIL import Image
        Image.fromarray(img).save(path, "PDF", resolution=150.0)
    else:
        cv2.imwrite(str(path), img)


def encode_png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def to_canonical(img):
    """Fit an arbitrary page onto the canonical canvas (letterboxed in white)."""
    h, w = img.shape[:2]
    s = min(PAGE_W / w, PAGE_H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    y0, x0 = (PAGE_H - nh) // 2, (PAGE_W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def tint(img):
    """Compress the tone range so white paper sits at PAPER_WHITE."""
    f = img.astype(np.float32) / 255.0
    return (INK_BLACK + f * (PAPER_WHITE - INK_BLACK)).round().astype(np.uint8)


def marker_boxes():
    """Top-left pixel position of each marker (ids 0..3 = TL, TR, BR, BL)."""
    m, s = MARKER_MARGIN, MARKER_SIZE
    return {
        0: (m, m),
        1: (PAGE_W - m - s, m),
        2: (PAGE_W - m - s, PAGE_H - m - s),
        3: (m, PAGE_H - m - s),
    }


def marker_corners():
    """Canonical corner coordinates (TL, TR, BR, BL order like ArUco)."""
    s = MARKER_SIZE
    return {i: np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], np.float32)
            for i, (x, y) in marker_boxes().items()}


def marker_exclusion_mask():
    """True where pixels belong to a fiducial plus its quiet zone."""
    mask = np.zeros((PAGE_H, PAGE_W), bool)
    q = MARKER_MARGIN
    for x, y in marker_boxes().values():
        mask[max(0, y - q):y + MARKER_SIZE + q, max(0, x - q):x + MARKER_SIZE + q] = True
    return mask


def add_fiducials(img):
    out = img.copy()
    for mid, (x, y) in marker_boxes().items():
        marker = cv2.aruco.generateImageMarker(_DICT, mid, MARKER_SIZE)
        q = MARKER_MARGIN // 2
        out[y - q:y + MARKER_SIZE + q, x - q:x + MARKER_SIZE + q] = 255
        out[y:y + MARKER_SIZE, x:x + MARKER_SIZE] = marker
    return out


def detect_markers(img):
    corners, ids, _ = _DETECTOR.detectMarkers(img)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            if 0 <= i <= 3 and i not in found:
                found[int(i)] = c.reshape(4, 2)
    if len(found) < 4:
        # Retry on a contrast-normalised copy (dim phone photos, glare)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
        corners, ids, _ = _DETECTOR.detectMarkers(clahe)
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                if 0 <= i <= 3 and int(i) not in found:
                    found[int(i)] = c.reshape(4, 2)
    return found


def rectify(img):
    """Warp a captured page back to the canonical frame using the fiducials.

    Returns (canonical_img, info). With >=3 markers a homography (or affine
    from 3) is fitted on all marker corners; otherwise img is returned
    unchanged and info["synced"] is False so blind search can take over.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = detect_markers(img)
    canon = marker_corners()
    info = {"markers": sorted(found), "synced": False}
    if len(found) >= 3:
        src = np.concatenate([found[i] for i in sorted(found)])
        dst = np.concatenate([canon[i] for i in sorted(found)])
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is not None:
            warped = cv2.warpPerspective(img, H, (PAGE_W, PAGE_H), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            info["synced"] = True
            return warped, info
    return img, info
