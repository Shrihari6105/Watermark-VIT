"""Master-key handling and keyed pseudo-randomness.

All secret randomness (block permutations, coefficient splits, Tardos biases,
fragile HMAC keys) is derived from a single master key with HMAC-SHA256, so
the investigator only needs that one key plus the registry.
"""
import hashlib
import hmac
import os
from pathlib import Path

import numpy as np


def load_or_create_key(path):
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    path.write_bytes(key)
    os.chmod(path, 0o600)
    return key


def subkey(master, label):
    return hmac.new(master, label.encode(), hashlib.sha256).digest()


def keyed_rng(master, label):
    seed = int.from_bytes(subkey(master, label)[:8], "big")
    return np.random.default_rng(seed)
