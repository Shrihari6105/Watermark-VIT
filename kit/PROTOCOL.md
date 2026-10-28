# WaterMark-VIT field capture protocol

1. Print `print_me.pdf` on plain A4/Letter paper, single-sided, with **"Fit to printable area" / "Shrink to fit"** so the four black corner markers are printed completely (scale does not matter; the markers correct for it). Check every corner marker is intact. Note the printer model.
2. On pages marked *PLEASE WRITE ON THIS PAGE*, write a few lines of answers in pen and fill some marks boxes, like a real student and evaluator would.
3. Capture each page through as many channels as you can (table below). One file per capture.
4. **File names:** `<page>_<channel>_<device>.<ext>`, e.g. `P03_phone30_pixel7.jpg`, `P08_scan300_hpm428.png`. Extra suffixes are fine: `P03_phone30_pixel7_2.jpg`.
5. Convert HEIC photos to JPG (iPhone: Settings -> Camera -> Formats -> Most Compatible).
6. Put all captures in one folder and run `python fieldkit.py analyze <kit> <folder>`.

Keep the kit's `--data` directory: it holds the key and registry needed to decode.

## Channels

| Channel tag | How to capture |
|---|---|
| `scan150` | Flatbed scan at 150 dpi (grayscale or colour) |
| `scan300` | Flatbed scan at 300 dpi |
| `phone0` | Phone photo straight above the page, ~40 cm, whole page in frame |
| `phone30` | Phone photo tilted ~30 degrees, ~40 cm |
| `phone45` | Phone photo tilted ~45 degrees, ~60 cm |
| `screen` | Open the page's PNG from digital/ full-screen on a laptop and photograph the screen |
| `whatsapp` | Send a phone0 photo to yourself on WhatsApp/Telegram and save the received image |
| `crop` | Photo or screenshot of only part of the page (no corner markers in frame) |

Record lighting and distance per device if you can; report them with the results.

## Pages

| Page | Type | Write on it | Suggested channels |
|---|---|---|---|
| P01 | booklet | yes | `scan150`, `scan300`, `phone0`, `phone30` |
| P02 | question | no | `scan300`, `phone0`, `phone30`, `phone45` |
| P03 | booklet | yes | `phone0`, `phone30`, `phone45`, `screen` |
| P04 | question | no | `phone30`, `phone45`, `screen`, `whatsapp` |
| P05 | booklet | yes | `phone45`, `screen`, `whatsapp`, `crop` |
| P06 | question | no | `screen`, `whatsapp`, `crop`, `scan150` |
| P07 | booklet | yes | `whatsapp`, `crop`, `scan150`, `scan300` |
| P08 | question | no | `crop`, `scan150`, `scan300`, `phone0` |
| P09 | booklet | yes | `scan150`, `scan300`, `phone0`, `phone30` |
| P10 | question | no | `scan300`, `phone0`, `phone30`, `phone45` |
| P11 | booklet | yes | `phone0`, `phone30`, `phone45`, `screen` |
| P12 | question | no | `phone30`, `phone45`, `screen`, `whatsapp` |

Minimum useful set: every page through `scan300`, `phone0` and `phone30` on at least two phones. That gives ≥ 36 physical captures for a 12-page kit.
