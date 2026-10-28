"""Fragile tamper-localisation layer (applied to the digitised, evaluated script).

The page is cut into 16x16 blocks. For block i the 256-bit tag
    HMAC-SHA256(k_script, block_index || 7 MSB planes of block i)
is written into the block's 256 LSBs, in a per-block order scrambled by a
keyed logistic (chaotic) map. Binding the block index and script ID into the
MAC defeats copy-paste/collage (VQ) attacks; scrambling hides the tag layout.
Any change to a block's MSBs (e.g. a "4" rewritten as "9" in the marks table)
invalidates exactly that block, giving a block-level tamper map.

The output must be stored losslessly (PNG); lossy re-encoding is itself
reported as tampering, which is the intended behaviour of a fragile mark.
"""
import hashlib
import hmac

import numpy as np

from .config import FRAGILE_BLOCK
from .keys import subkey

FB = FRAGILE_BLOCK


def _logistic_perm(key, n=FB * FB, r=3.99):
    x = 0.1 + 0.8 * (int.from_bytes(key[:8], "big") / 2 ** 64)
    for _ in range(100):                     # burn-in
        x = r * x * (1 - x)
    seq = np.empty(n)
    for i in range(n):
        x = r * x * (1 - x)
        seq[i] = x
    return np.argsort(seq)


def _blocks(img):
    h, w = img.shape
    R, C = h // FB, w // FB
    return img[:R * FB, :C * FB].reshape(R, FB, C, FB).transpose(0, 2, 1, 3), R, C


def _tags(img, master, script_id):
    key = subkey(master, f"fragile/{script_id}")
    blocks, R, C = _blocks(img & 0xFE)
    tags = np.empty((R, C, FB * FB), np.uint8)
    for r in range(R):
        for c in range(C):
            msg = (r * C + c).to_bytes(4, "big") + blocks[r, c].tobytes()
            digest = hmac.new(key, msg, hashlib.sha256).digest()
            tags[r, c] = np.unpackbits(np.frombuffer(digest, np.uint8))
    perm = _logistic_perm(subkey(master, f"fragile-perm/{script_id}"))
    placed = np.empty_like(tags)
    placed[..., perm] = tags
    return placed.reshape(R, C, FB, FB), R, C


def seal(img, master, script_id):
    """Embed the fragile layer. `img` must already be canonical uint8."""
    tags, R, C = _tags(img, master, script_id)
    out = img.copy()
    view = out[:R * FB, :C * FB].reshape(R, FB, C, FB).transpose(0, 2, 1, 3)
    view[...] = (view & 0xFE) | tags
    return out


def verify(img, master, script_id):
    """Return a boolean tamper map [R, C] (True = block failed authentication)."""
    tags, R, C = _tags(img, master, script_id)
    blocks, _, _ = _blocks(img)
    lsb = blocks & 1
    mismatched = (lsb != tags).reshape(R, C, -1).sum(axis=2)
    return mismatched > 0


def heatmap_overlay(img, tamper_map):
    """RGB visualisation with tampered blocks tinted red."""
    import cv2
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    mask = np.kron(tamper_map, np.ones((FB, FB), bool))
    h, w = mask.shape
    overlay = rgb[:h, :w]
    overlay[mask] = overlay[mask] * 0.45 + np.array([40, 40, 255]) * 0.55
    grid = np.kron(tamper_map, np.ones((FB, FB)))  # outline tampered region
    edges = cv2.Canny((grid * 255).astype(np.uint8), 50, 150) > 0
    overlay[edges] = (0, 0, 255)
    return rgb.round().astype(np.uint8)
