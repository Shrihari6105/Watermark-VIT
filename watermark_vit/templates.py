"""Synthetic VIT-style question papers / answer booklets and simulated handwriting."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import PAGE_H, PAGE_W

QUESTIONS = [
    "Explain the working of a DWT-based image compression pipeline.",
    "Derive the time complexity of Dijkstra's algorithm with a binary heap.",
    "Describe the CIA triad and give one attack against each property.",
    "Compare TCP and UDP congestion behaviour with suitable diagrams.",
    "What is quantisation index modulation? State its robustness trade-off.",
    "Design a 4-bit synchronous counter using JK flip-flops.",
    "Explain normalisation up to BCNF with an example relation.",
    "Discuss the role of the Fourier transform in signal denoising.",
    "Write an algorithm to detect a cycle in a directed graph.",
    "Explain public-key infrastructure and certificate revocation.",
    "State and prove the master theorem for divide-and-conquer recurrences.",
    "Describe how a hash-based message authentication code works.",
]
COURSES = ["CSE3501 Information Security", "BCSE302L Database Systems",
           "BECE204L Microprocessors", "BCSE306L Artificial Intelligence",
           "BMAT205L Discrete Mathematics"]

# Marks table geometry (used by the tamper attacks to target the marks boxes)
MARKS_TOP = 330
MARKS_LEFT = 140
MARKS_CELL_W = 90
MARKS_CELL_H = 60
MARKS_COLS = 10


def _font(size):
    return ImageFont.load_default(size=size)


def answer_sheet(seed=0, kind="booklet"):
    """Render a blank exam page (white background, no watermark, no fiducials)."""
    rng = np.random.default_rng(seed)
    im = Image.new("L", (PAGE_W, PAGE_H), 255)
    d = ImageDraw.Draw(im)
    course = COURSES[seed % len(COURSES)]
    d.text((PAGE_W // 2, 120), "VELLORE INSTITUTE OF TECHNOLOGY", font=_font(38), fill=0, anchor="mm")
    title = "ANSWER BOOKLET" if kind == "booklet" else "QUESTION PAPER"
    d.text((PAGE_W // 2, 170), f"{title}  -  {course}", font=_font(26), fill=0, anchor="mm")
    d.text((PAGE_W // 2, 205), f"Continuous Assessment Test {1 + seed % 2}   |   Slot {'ABCDEFG'[seed % 7]}1   |   Max marks: 50",
           font=_font(20), fill=40, anchor="mm")
    d.line((120, 235, PAGE_W - 120, 235), fill=0, width=3)

    # Registration-number boxes
    d.text((140, 262), "Reg. No.", font=_font(22), fill=0)
    for i in range(9):
        x = 270 + i * 44
        d.rectangle((x, 255, x + 38, 295), outline=0, width=2)
    d.text((740, 262), "Invigilator sign:", font=_font(22), fill=0)
    d.line((925, 290, 1140, 290), fill=0, width=2)

    # Marks table
    top, left, cw, ch = MARKS_TOP, MARKS_LEFT, MARKS_CELL_W, MARKS_CELL_H
    for c in range(MARKS_COLS + 1):
        d.line((left + c * cw, top, left + c * cw, top + 2 * ch), fill=0, width=2)
    for r in range(3):
        d.line((left, top + r * ch, left + MARKS_COLS * cw, top + r * ch), fill=0, width=2)
    for c in range(MARKS_COLS):
        label = f"Q{c + 1}" if c < MARKS_COLS - 1 else "Total"
        d.text((left + c * cw + cw // 2, top + ch // 2), label, font=_font(22), fill=0, anchor="mm")

    if kind == "booklet":
        y = 500
        d.text((140, y - 40), "Answers (write within the margins)", font=_font(22), fill=60)
        while y < PAGE_H - 140:
            d.line((140, y, PAGE_W - 140, y), fill=170, width=1)
            y += 46
        d.line((200, 470, 200, PAGE_H - 140), fill=120, width=2)   # margin rule
    else:
        y = 500
        for qn, q in enumerate(rng.choice(QUESTIONS, size=8, replace=False), 1):
            marks = int(rng.choice([5, 6, 8, 10]))
            d.text((140, y), f"{qn}.", font=_font(24), fill=0)
            d.text((190, y), q, font=_font(24), fill=0)
            d.text((PAGE_W - 140, y), f"[{marks}]", font=_font(24), fill=0, anchor="ra")
            y += 48
            for _ in range(int(rng.integers(1, 3))):
                words = " ".join(rng.choice(["Justify", "your", "answer", "with", "a", "neat",
                                             "diagram", "and", "example", "where", "relevant"], 9))
                d.text((190, y), words.capitalize() + ".", font=_font(20), fill=60)
                y += 34
            y += 40
    d.text((PAGE_W // 2, PAGE_H - 100), f"Page 1   -   template #{seed}", font=_font(18), fill=90, anchor="mm")
    return np.array(im)


def add_handwriting(img, seed=0, marks=None, density=0.6, ink=35):
    """Simulate a student writing in the answer area and an evaluator filling marks.

    Strokes are drawn *after* watermark embedding, mirroring embed -> print ->
    write -> scan. Returns a new image.
    """
    rng = np.random.default_rng(seed)
    out = img.copy()
    y = 500 - 10
    while y < PAGE_H - 160:
        if rng.random() < density:
            x = 220
            end = int(rng.integers(600, PAGE_W - 160))
            while x < end:
                w = int(rng.integers(30, 110))
                pts = []
                for t in np.linspace(0, 1, max(6, w // 6)):
                    px = x + t * w
                    py = y - 14 + 10 * np.sin(t * rng.uniform(6, 14) + rng.uniform(0, 6)) + rng.normal(0, 2)
                    pts.append((px, py))
                cv2.polylines(out, [np.int32(pts)], False, ink, int(rng.integers(2, 4)), cv2.LINE_AA)
                x += w + int(rng.integers(12, 28))
        y += 46
    if marks is not None:
        write_marks(out, marks)
    return out


def write_marks(img, marks, ink=30):
    """Write evaluator marks into the marks table (in place)."""
    for c, m in enumerate(marks[:MARKS_COLS]):
        x = MARKS_LEFT + c * MARKS_CELL_W + 22
        y = MARKS_TOP + MARKS_CELL_H + 45
        cv2.putText(img, str(m), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.3, ink, 2, cv2.LINE_AA)
    return img


def marks_cell(col):
    """(x0, y0, x1, y1) of the marks cell for question column `col`."""
    x0 = MARKS_LEFT + col * MARKS_CELL_W
    y0 = MARKS_TOP + MARKS_CELL_H
    return x0, y0, x0 + MARKS_CELL_W, y0 + MARKS_CELL_H
