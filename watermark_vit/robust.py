"""Robust attribution layer: DWT -> block DCT -> differential coefficient modulation.

Each 32x32 pixel block carries one bit. The block's Haar LL band (16x16) is
DCT-transformed and 8 low/mid-frequency coefficients are split (by a keyed
choice) into groups A and B. Bit b is written by forcing
    s * (mean(C_A) - mean(C_B)) >= T,   s = +1 if b else -1
with the minimum-energy correction (capped at DMAX on text-heavy blocks).
The sign test is invariant to gain and offset changes, which is what
print-scan and camera capture mostly do to the tone curve.

Block roles form a checkerboard:
  * even (bx+by): a periodic tile of 16x16 blocks whose 128 even positions
    hold the 64-bit RS-coded ID and a 64-bit keyed *pilot* sequence. The
    pilot is identical on every copy, so its correlation tells whether a
    watermark is present at all and, in a crop without fiducials, which
    alignment/tile phase is the right one;
  * odd  (bx+by): the Tardos fingerprint, spread over the whole page by a
    keyed interleaver.
"""
import numpy as np
import pywt
from scipy.fft import dctn, idctn

from .config import (ID_TILE_COLS, ID_TILE_ROWS, PAGE_H, PAGE_W, ROBUST_BLOCK,
                     ROBUST_DMAX, ROBUST_T)
from .ecc import CODE_BITS
from .imaging import marker_exclusion_mask
from .keys import keyed_rng

LB = ROBUST_BLOCK // 2                         # LL sub-block size (16)
ROWS, COLS = PAGE_H // ROBUST_BLOCK, PAGE_W // ROBUST_BLOCK
CANDIDATES = [(0, 2), (2, 0), (1, 2), (2, 1), (0, 3), (3, 0), (1, 3), (3, 1)]
N_TILE = ID_TILE_COLS * ID_TILE_ROWS          # 256 tile positions
ROLE_ID, ROLE_PILOT, ROLE_FP = 0, 1, 2
PILOT_BITS = N_TILE // 2 - CODE_BITS          # 64


def _tile_pos(bx, by):
    return (by % ID_TILE_ROWS) * ID_TILE_COLS + (bx % ID_TILE_COLS)


class Layout:
    """Keyed block -> bit assignment, shared by embedder and extractor."""

    def __init__(self, master, tardos_len):
        rng = keyed_rng(master, "robust-layout")
        # Coefficient split per tile position: weight vector over the 16x16 DCT
        self.weights = np.zeros((N_TILE, LB * LB), np.float32)
        for k in range(N_TILE):
            order = rng.permutation(len(CANDIDATES))
            for n, ci in enumerate(order):
                u, v = CANDIDATES[ci]
                self.weights[k, u * LB + v] = 0.25 if n < 4 else -0.25

        # Role and slot of each tile position (even -> ID or pilot, odd -> Tardos)
        ty, tx = np.divmod(np.arange(N_TILE), ID_TILE_COLS)
        even = np.flatnonzero((tx + ty) % 2 == 0)
        even = even[rng.permutation(len(even))]
        self.tile_role = np.full(N_TILE, ROLE_FP)
        self.tile_slot = np.full(N_TILE, -1)
        self.tile_role[even[:CODE_BITS]] = ROLE_ID
        self.tile_slot[even[:CODE_BITS]] = np.arange(CODE_BITS)
        self.tile_role[even[CODE_BITS:]] = ROLE_PILOT
        self.tile_slot[even[CODE_BITS:]] = np.arange(PILOT_BITS)
        self.pilot = rng.integers(0, 2, PILOT_BITS)

        by, bx = np.mgrid[0:ROWS, 0:COLS]
        excl = marker_exclusion_mask()
        usable = ~excl.reshape(ROWS, ROBUST_BLOCK, COLS, ROBUST_BLOCK).any(axis=(1, 3))
        self.tile_pos = _tile_pos(bx, by)
        role = self.tile_role[self.tile_pos]
        self.is_id = usable & (role == ROLE_ID)
        self.is_pilot = usable & (role == ROLE_PILOT)
        self.slot = self.tile_slot[self.tile_pos]

        self.is_fp = usable & (role == ROLE_FP)
        self.fp_bit = np.full((ROWS, COLS), -1)
        if tardos_len:
            idx = np.flatnonzero(self.is_fp)
            idx = idx[rng.permutation(len(idx))]
            self.fp_bit.flat[idx] = np.arange(len(idx)) % tardos_len
        self.tardos_len = tardos_len


def _ll_dct(img):
    """Return (LL-DCT block tensor [R,C,256], LL band, detail bands) of a region."""
    cA, details = pywt.dwt2(img.astype(np.float64), "haar")
    R, C = cA.shape[0] // LB, cA.shape[1] // LB
    blocks = cA[:R * LB, :C * LB].reshape(R, LB, C, LB).transpose(0, 2, 1, 3)
    coefs = dctn(blocks, axes=(2, 3), norm="ortho")
    return coefs.reshape(R, C, LB * LB), cA, details


def embed(img, master, id_bits, fp_bits, strength=ROBUST_T):
    """Embed the ID codeword, pilot and Tardos codeword into a canonical page."""
    layout = Layout(master, len(fp_bits))
    coefs, cA, details = _ll_dct(img)
    W = layout.weights[layout.tile_pos]                         # [R,C,256]
    d = np.einsum("rck,rck->rc", coefs, W)

    target = np.zeros((ROWS, COLS))
    target[layout.is_id] = np.asarray(id_bits)[layout.slot[layout.is_id]]
    target[layout.is_pilot] = layout.pilot[layout.slot[layout.is_pilot]]
    active = layout.is_id | layout.is_pilot
    if len(fp_bits):
        target[layout.is_fp] = np.asarray(fp_bits)[layout.fp_bit[layout.is_fp]]
        active |= layout.is_fp
    s = 2 * target - 1
    delta = np.clip(strength - s * d, 0, ROBUST_DMAX) * active
    coefs = coefs + (2 * s * delta)[..., None] * W

    blocks = idctn(coefs.reshape(ROWS, COLS, LB, LB), axes=(2, 3), norm="ortho")
    cA = blocks.transpose(0, 2, 1, 3).reshape(ROWS * LB, COLS * LB)
    out = pywt.idwt2((cA, details), "haar")
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _soft_blocks(d):
    return np.clip(d / ROBUST_T, -1.5, 1.5)


def _aggregate(values, bit_index, n_bits):
    sums = np.bincount(bit_index, weights=values, minlength=n_bits)
    counts = np.bincount(bit_index, minlength=n_bits)
    return np.divide(sums, np.maximum(counts, 1)), counts


def _pilot_corr(soft, slots, pilot):
    if len(soft) == 0:
        return 0.0
    return float(np.mean(np.clip(soft, -1, 1) * (2 * pilot[slots] - 1)))


def extract(img, master, tardos_len):
    """Soft extraction from a page already in the canonical frame."""
    layout = Layout(master, tardos_len)
    coefs, _, _ = _ll_dct(img)
    d = np.einsum("rck,rck->rc", coefs, layout.weights[layout.tile_pos])
    soft = _soft_blocks(d)
    id_soft, _ = _aggregate(soft[layout.is_id], layout.slot[layout.is_id], CODE_BITS)
    pilot = _pilot_corr(soft[layout.is_pilot], layout.slot[layout.is_pilot], layout.pilot)
    fp_soft = np.zeros(tardos_len)
    if tardos_len:
        fp_soft, _ = _aggregate(soft[layout.is_fp], layout.fp_bit[layout.is_fp], tardos_len)
    return {"id_soft": id_soft, "pilot": pilot, "fp_soft": fp_soft, "block_soft": soft}


def _phase_scan(d_all, layout):
    """Score every tile phase for one pixel alignment; return best (pilot, ky, kx)."""
    R, C, _ = d_all.shape
    i, j = np.mgrid[0:R, 0:C]
    best = (-np.inf, 0, 0)
    for ky in range(ID_TILE_ROWS):
        for kx in range(ID_TILE_COLS):
            tp = _tile_pos(j + kx, i + ky)
            sel = layout.tile_role[tp] == ROLE_PILOT
            d = np.take_along_axis(d_all, tp[..., None], axis=2)[..., 0]
            pc = _pilot_corr(_soft_blocks(d[sel]), layout.tile_slot[tp[sel]], layout.pilot)
            if pc > best[0]:
                best = (pc, ky, kx)
    return best


def _decode_at(img, layout, ax, ay, ky, kx):
    sub = img[ay:, ax:]
    R, C = sub.shape[0] // ROBUST_BLOCK, sub.shape[1] // ROBUST_BLOCK
    coefs, _, _ = _ll_dct(sub[:R * ROBUST_BLOCK, :C * ROBUST_BLOCK])
    i, j = np.mgrid[0:R, 0:C]
    tp = _tile_pos(j + kx, i + ky)
    d = np.einsum("rck,rck->rc", coefs, layout.weights[tp])
    soft = _soft_blocks(d)
    role, slot = layout.tile_role[tp], layout.tile_slot[tp]
    id_soft, _ = _aggregate(soft[role == ROLE_ID], slot[role == ROLE_ID], CODE_BITS)
    pilot = _pilot_corr(soft[role == ROLE_PILOT], slot[role == ROLE_PILOT], layout.pilot)
    return pilot, id_soft


def blind_extract(img, master, step=4):
    """Recover the ID from a crop with unknown offset (scale assumed ~1).

    Coarse search over pixel alignment (mod 32, `step` px) and all 256 tile
    phases, scored by pilot correlation; then a 1-px refinement around the
    winner. Returns dict(pilot, id_soft, hypothesis).
    """
    layout = Layout(master, 0)
    Wt = layout.weights.T
    if min(img.shape) < 4 * ROBUST_BLOCK:
        return {"pilot": 0.0, "id_soft": np.zeros(CODE_BITS), "hypothesis": None}
    best = (-np.inf, None)
    for ay in range(0, ROBUST_BLOCK, step):
        for ax in range(0, ROBUST_BLOCK, step):
            sub = img[ay:, ax:]
            R, C = sub.shape[0] // ROBUST_BLOCK, sub.shape[1] // ROBUST_BLOCK
            coefs, _, _ = _ll_dct(sub[:R * ROBUST_BLOCK, :C * ROBUST_BLOCK])
            pc, ky, kx = _phase_scan(coefs @ Wt, layout)
            if pc > best[0]:
                best = (pc, (ax, ay, ky, kx))
    ax, ay, ky, kx = best[1]
    result = None
    for dy in range(-(step // 2), step // 2 + 1):
        for dx in range(-(step // 2), step // 2 + 1):
            nx, ny = ax + dx, ay + dy
            kx2, ky2 = kx, ky
            if nx < 0:                       # origin wraps to the next block
                nx, kx2 = nx + ROBUST_BLOCK, kx + 1
            if ny < 0:
                ny, ky2 = ny + ROBUST_BLOCK, ky + 1
            pilot, id_soft = _decode_at(img, layout, nx, ny, ky2 % ID_TILE_ROWS, kx2 % ID_TILE_COLS)
            if result is None or pilot > result["pilot"]:
                result = {"pilot": pilot, "id_soft": id_soft, "hypothesis": (nx, ny, ky2, kx2)}
    return result
