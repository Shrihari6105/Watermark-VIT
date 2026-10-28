# WaterMark-VIT evaluation report

Pages: 16 (half answer booklets, half question papers) | Class size n = 200 | Tardos c = 2, eps = 1e-3

Physical channels are **simulated** (see `watermark_vit/attacks.py`); real printer/phone captures are still required for the final paper.

## Imperceptibility

| Metric | Mean ± 95% CI |
|---|---|
| PSNR | 41.118 ± 0.174 |
| SSIM | 0.983 ± 0.000 |

## Robustness (leak attribution)

`hand` = handwriting + marks added after embedding (embed → write → leak).

| Channel | ID acc. (all) | ID acc. clean | ID acc. handwritten | Wrong ID | Raw BER | Tardos catches leaker |
|---|---|---|---|---|---|---|
| none | 100.0% | 100.0% | 100.0% | 0.0% | 0.025 | 100.0% |
| jpeg90 | 100.0% | 100.0% | 100.0% | 0.0% | 0.025 | 100.0% |
| jpeg70 | 100.0% | 100.0% | 100.0% | 0.0% | 0.026 | 100.0% |
| jpeg50 | 100.0% | 100.0% | 100.0% | 0.0% | 0.027 | 100.0% |
| jpeg30 | 100.0% | 100.0% | 100.0% | 0.0% | 0.042 | 100.0% |
| webp60 | 100.0% | 100.0% | 100.0% | 0.0% | 0.063 | 100.0% |
| resize0.5 | 100.0% | 100.0% | 100.0% | 0.0% | 0.024 | 100.0% |
| resize1.5 | 100.0% | 100.0% | 100.0% | 0.0% | 0.025 | 100.0% |
| rotate3 | 100.0% | 100.0% | 100.0% | 0.0% | 0.024 | 100.0% |
| rotate-5 | 100.0% | 100.0% | 100.0% | 0.0% | 0.025 | 100.0% |
| noise8 | 100.0% | 100.0% | 100.0% | 0.0% | 0.038 | 100.0% |
| blur5 | 100.0% | 100.0% | 100.0% | 0.0% | 0.028 | 100.0% |
| stirmark | 95.8% | 100.0% | 87.5% | 0.0% | 0.080 | 100.0% |
| whatsapp | 100.0% | 100.0% | 100.0% | 0.0% | 0.031 | 100.0% |
| screenshot1080 | 100.0% | 100.0% | 100.0% | 0.0% | 0.024 | 100.0% |
| crop50 | 95.8% | 100.0% | 87.5% | 0.0% | – | – |
| crop30 | 83.3% | 100.0% | 50.0% | 0.0% | – | – |
| printscan150 | 100.0% | 100.0% | 100.0% | 0.0% | 0.038 | 100.0% |
| printscan300 | 100.0% | 100.0% | 100.0% | 0.0% | 0.044 | 100.0% |
| phone0deg40cm | 100.0% | 100.0% | 100.0% | 0.0% | 0.057 | 100.0% |
| phone30deg40cm | 100.0% | 100.0% | 100.0% | 0.0% | 0.057 | 100.0% |
| phone45deg60cm | 100.0% | 100.0% | 100.0% | 0.0% | 0.062 | 100.0% |
| screenshoot | 100.0% | 100.0% | 100.0% | 0.0% | 0.056 | 100.0% |

## Attribution safety

- Innocent Tardos trials: **100296**, innocents accused: **0** (empirical rate 0.00e+00; design bound eps/n per trial)
- Unwatermarked pages decoded or accused: **0 / 80**

## Collusion (c = 2)

| Attack | ≥1 colluder accused | Both accused | Top-2 = colluders | Innocents accused / trial |
|---|---|---|---|---|
| average+digital | 100.0% | 93.8% | 100.0% | 0.00 |
| average+jpeg70 | 100.0% | 93.8% | 100.0% | 0.00 |
| average+printscan300 | 100.0% | 100.0% | 100.0% | 0.00 |
| splice+digital | 100.0% | 75.0% | 100.0% | 0.00 |
| splice+jpeg70 | 100.0% | 75.0% | 100.0% | 0.00 |
| splice+printscan300 | 100.0% | 68.8% | 100.0% | 0.00 |
| interleave+digital | 100.0% | 100.0% | 100.0% | 0.00 |
| interleave+jpeg70 | 100.0% | 93.8% | 100.0% | 0.00 |
| interleave+printscan300 | 93.8% | 93.8% | 100.0% | 0.00 |

## Tamper localisation (fragile layer, 16×16 blocks)

| Attack | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| marks_overwrite | 1.000 | 1.000 | 1.000 | 1.000 |
| copy_move | 1.000 | 1.000 | 1.000 | 1.000 |
| inpaint | 1.000 | 1.000 | 1.000 | 1.000 |
| erase | 1.000 | 1.000 | 1.000 | 1.000 |

- False alarms on untouched sealed scripts (PNG round trip): 0.0%
- Fraction of blocks flagged after a JPEG Q95 re-save: 100.0% (expected: the fragile layer rejects lossy copies)

## Timing (CPU, seconds per page)

| Step | Mean ± 95% CI |
|---|---|
| embed | 0.078 ± 0.003 |
| extract | 0.165 ± 0.026 |
| seal | 0.079 ± 0.008 |
