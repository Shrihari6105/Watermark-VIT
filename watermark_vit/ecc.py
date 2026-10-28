"""Payload packing and Reed-Solomon protection.

Payload = pseudonymous roster index (10 bits) || exam code (6 bits) = 2 bytes.
RS(8, 2) over GF(256) adds 6 parity bytes -> 64-bit codeword that corrects up
to 3 byte errors, or up to 4 byte erasures (bytes whose soft confidence is low).
"""
import numpy as np
from reedsolo import ReedSolomonError, RSCodec

from .config import EXAM_BITS, ROSTER_BITS, RS_NSYM

_rs = RSCodec(RS_NSYM)
CODE_BITS = (2 + RS_NSYM) * 8
MAX_ERASURES = RS_NSYM - 2


def pack(roster_idx, exam_code):
    if not 0 <= roster_idx < (1 << ROSTER_BITS):
        raise ValueError("roster index out of range")
    if not 0 <= exam_code < (1 << EXAM_BITS):
        raise ValueError("exam code out of range")
    value = (roster_idx << EXAM_BITS) | exam_code
    return value.to_bytes(2, "big")


def unpack(data):
    value = int.from_bytes(bytes(data[:2]), "big")
    return value >> EXAM_BITS, value & ((1 << EXAM_BITS) - 1)


def encode(roster_idx, exam_code):
    """Return the 64-bit codeword as a uint8 array of 0/1."""
    cw = bytes(_rs.encode(pack(roster_idx, exam_code)))
    return np.unpackbits(np.frombuffer(cw, dtype=np.uint8))


def decode_soft(soft):
    """Decode from soft bit values (positive -> 1). Returns dict or None.

    The least reliable bytes are tried as erasures when plain decoding fails,
    which roughly doubles the correctable damage.
    """
    soft = np.asarray(soft, dtype=float)
    bits = (soft > 0).astype(np.uint8)
    data = bytes(np.packbits(bits))
    reliability = np.abs(soft).reshape(-1, 8).min(axis=1)
    order = list(np.argsort(reliability))
    # Keep >= 2 parity symbols of slack so noise cannot be "decoded" by erasing everything
    attempts = [[]] + [sorted(order[:k]) for k in range(1, MAX_ERASURES + 1)]
    for erase in attempts:
        try:
            msg, full, errata = _rs.decode(data, erase_pos=erase or None)
        except ReedSolomonError:
            continue
        roster_idx, exam_code = unpack(msg)
        corrected = np.unpackbits(np.frombuffer(bytes(full), dtype=np.uint8))
        return {
            "roster_idx": roster_idx,
            "exam_code": exam_code,
            "bit_errors": int(np.sum(corrected != bits)),
            "erasures": len(erase),
            "codeword": corrected,
        }
    return None
