"""Symmetric Tardos fingerprinting codes (Tardos 2003 / Skoric et al. 2008).

Each student j gets a binary codeword X_j of length L. Position biases p_i are
drawn from the arcsine distribution with cutoff t = 1/(300c). Given a pirate
word y (possibly with erasures), every student receives an accusation score;
scores of innocents are ~N(0, L), so the threshold
Z = sqrt(L_eff) * Phi^-1(1 - eps/n) bounds the false-accusation probability
by eps (Gaussian approximation; the evaluation script checks it empirically).
"""
import numpy as np
from scipy.stats import norm

from .keys import keyed_rng


def biases(master, exam_label, length, c):
    rng = keyed_rng(master, f"tardos-p/{exam_label}")
    t = 1.0 / (300 * c)
    t0 = np.arcsin(np.sqrt(t))
    r = rng.uniform(t0, np.pi / 2 - t0, size=length)
    return np.sin(r) ** 2


def codebook(master, exam_label, n_users, length, c):
    p = biases(master, exam_label, length, c)
    rng = keyed_rng(master, f"tardos-X/{exam_label}")
    X = (rng.random((n_users, length)) < p).astype(np.uint8)
    return X, p


def scores(X, p, soft, erasure_thresh=0.25):
    """Symmetric Tardos scores. `soft` > 0 means pirate bit 1; |soft| below
    erasure_thresh is treated as an erased (uninformative) position."""
    soft = np.asarray(soft, dtype=float)
    valid = np.abs(soft) >= erasure_thresh
    y = (soft > 0).astype(np.uint8)
    g1 = np.sqrt((1 - p) / p)        # reward for matching a 1
    g0 = np.sqrt(p / (1 - p))        # reward for matching a 0
    match_reward = np.where(y == 1, g1, g0)
    mismatch_pen = np.where(y == 1, -g0, -g1)
    U = np.where(X == y[None, :], match_reward[None, :], mismatch_pen[None, :])
    U = U * valid[None, :]
    return U.sum(axis=1), int(valid.sum())


def threshold(n_users, effective_len, eps):
    if effective_len == 0:
        return np.inf
    return float(np.sqrt(effective_len) * norm.ppf(1 - eps / max(n_users, 1)))
