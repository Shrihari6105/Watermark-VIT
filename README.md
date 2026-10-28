# WaterMark-VIT

WaterMark-VIT is a hidden watermark for exam papers. It has two jobs:

1. **Find who leaked a paper.** Every student's copy carries an invisible ID. If a photo, screenshot or scan of the paper turns up on WhatsApp, the program reads the ID and names the student.
2. **Catch edited marks.** After evaluation, a scanned answer script is sealed. If anyone later changes a mark, the program highlights the edited spot.

The idea comes from EditGuard (CVPR 2024), adapted to exam pages. It runs on a normal laptop, with no GPU and no training.

## Setup (once)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Try it

```bash
.venv/bin/python cli.py test my_assignment.pdf    # use any scanned pages you have
```

The command does three things:

1. It gives each page a different student's hidden ID.
2. It leaks each page 15 ways: WhatsApp, screenshot, JPEG, crop, phone photo, photo of a screen, print-scan, and more.
3. It works out who leaked each copy.

It prints the result and writes `test_output/report.md`.

To add a real capture, open a page from `test_output/pages/` on your laptop, photograph the screen, and save the photo as `test_output/leaks/P01_screen_myphone.jpg`. Then run `.venv/bin/python cli.py analyze`.

Other ways in:

```bash
.venv/bin/python cli.py demo         # the full story on a sample page: issue -> leak -> trace -> seal -> catch an edited mark
.venv/bin/python app.py              # web dashboard at http://127.0.0.1:5050
.venv/bin/python evaluate.py         # full evaluation -> results/report.md
.venv/bin/python -m pytest -q tests  # 15 tests
```

## Using it for a real exam

```bash
.venv/bin/python cli.py create-exam "CAT1 CSE3501" roster.txt          # one registration number per line
.venv/bin/python cli.py issue 1 issued/ --page question_paper.pdf      # one watermarked copy per student
.venv/bin/python cli.py investigate leaked_photo.jpg                   # who leaked it?
.venv/bin/python cli.py seal evaluated_scan.png sealed.png             # after marking
.venv/bin/python cli.py verify sealed.png --overlay heatmap.png        # were the marks changed?
```

Keep the `data/` folder safe. It holds the secret key and the student list, and nothing can be decoded without it.

## How it works

| Part | Plain idea | File |
|---|---|---|
| Corner markers | Four small black squares let any tilted photo be straightened back into a flat page. | `imaging.py` |
| Hidden ID | The page is split into small blocks. Each block is nudged very slightly to store one bit, in a way that survives JPEG compression, blur and camera tone changes. | `robust.py` |
| Error correction | Reed–Solomon coding fixes damaged bits. A statistical test makes sure noise never names a student. | `ecc.py`, `pipeline.py` |
| Fingerprint | A second, longer code (Tardos) still catches students who combine their copies to hide. | `tardos.py` |
| Tamper seal | A cryptographic hash (HMAC) of each 16×16 block is hidden in that block. Any edit breaks it and shows on a heat-map. | `fragile.py` |
| Registry | A database maps hidden numbers to students, so only an index is embedded, never the registration number. | `registry.py` |
| Leak simulator | Computer models of WhatsApp, screenshots, phone photos, print-scan and more. | `attacks.py` |
| Leak test | The `cli.py test` command. | `leaktest.py` |

## Results

Full evaluation (16 pages, 200 students; `results/report.md`):

- The watermark is invisible: PSNR ≈ 41 dB, SSIM ≈ 0.98.
- The right student was found 100% of the time for JPEG, WhatsApp, screenshots, print-scan, phone photos at 0–45° and photos of a screen. The weakest cases are small crops of handwritten pages.
- A wrong student was never named, and unmarked pages were never flagged.
- When two students combine their copies, at least one is caught in 94–100% of trials, and no innocent student is accused.
- Edited marks are located exactly (F1 = 1.0).

On six real handwritten, phone-scanned assignment pages, `cli.py test` traced **90/90** leaked copies to the right student, with 0 wrong. The original unwatermarked pages all returned "No watermark detected".

## Limitations

- The leaks are **computer-simulated**. Real phone and screen captures are still the strongest proof, and you can add your own as described above.
- The tamper seal only works on lossless PNG copies. Re-saving as JPEG counts as tampering.
- The fingerprint is designed for up to 2 colluding students.
- The `data/` folder is sensitive. Get faculty approval before using this on real students.
