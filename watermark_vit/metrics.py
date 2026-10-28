"""Imperceptibility, robustness and localisation metrics."""
import cv2
import numpy as np


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def ssim(a, b):
    """Gaussian-window SSIM (Wang et al. 2004)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = g(a), g(b)
    va, vb, cov = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2, g(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(m.mean())


def ber(soft, bits):
    return float(np.mean((np.asarray(soft) > 0).astype(np.uint8) != np.asarray(bits)))


def nc(soft, bits):
    """Normalised correlation between bipolar embedded bits and extracted soft values."""
    s, b = np.asarray(soft, float), 2.0 * np.asarray(bits, float) - 1
    return float(np.dot(s, b) / (np.linalg.norm(s) * np.linalg.norm(b) + 1e-12))


def block_truth(original, tampered, block=16):
    """Ground-truth tampered blocks: any block with at least one modified pixel."""
    diff = original != tampered
    h, w = diff.shape
    return diff[:h // block * block, :w // block * block].reshape(
        h // block, block, w // block, block).any(axis=(1, 3))


def prf(pred, truth):
    tp = np.sum(pred & truth); fp = np.sum(pred & ~truth); fn = np.sum(~pred & truth)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return float(p), float(r), float(f), float(iou)
