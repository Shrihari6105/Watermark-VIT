"""Global constants for WaterMark-VIT.

Every page is processed on a fixed canonical canvas (A4 portrait at ~150 dpi,
rounded so the 32 px robust grid and 16 px fragile grid tile it exactly).
"""

# Canonical page geometry
PAGE_W = 1280
PAGE_H = 1792

# Paper tint: white is mapped to PAPER_WHITE and black to INK_BLACK so the
# robust watermark has headroom on both sides instead of clipping at 255.
PAPER_WHITE = 245
INK_BLACK = 10

# Corner fiducials (ArUco DICT_4X4_50, ids 0..3 = TL, TR, BR, BL)
MARKER_SIZE = 64
MARKER_MARGIN = 16

# Robust layer (DWT -> DCT -> differential coefficient modulation)
ROBUST_BLOCK = 32                 # pixel block; LL sub-block is 16x16
ROBUST_T = 40.0                   # target margin |mean(A) - mean(B)| in LL-DCT units
ROBUST_DMAX = 110.0               # cap on per-block correction (limits damage on text)
ID_TILE_COLS = 16                 # tile = 16 x 16 blocks; its 128 even positions hold
ID_TILE_ROWS = 16                 # the 64-bit ID codeword + a 64-bit keyed pilot

# Detection: minimum pilot correlation to accept that a watermark is present
PILOT_MIN = 0.25
PILOT_MIN_BLIND = 0.45            # blind search maximises over ~16k hypotheses -> higher bar

# Payload
ROSTER_BITS = 10                  # up to 1024 students per exam
EXAM_BITS = 6                     # 64 exam/session codes
RS_NSYM = 6                       # Reed-Solomon parity bytes (2 data + 6 parity = 64 bits)

# Tardos fingerprint
TARDOS_LEN = 320
TARDOS_C = 2
TARDOS_EPS = 1e-3

# Fragile layer
FRAGILE_BLOCK = 16                # 16x16 = 256 LSBs = one HMAC-SHA256 tag
