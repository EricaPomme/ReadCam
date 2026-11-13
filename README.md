# ReadCam

Live camera scanner that detects QR codes, barcodes, and text in real time.

![screenshot placeholder](https://img.shields.io/badge/status-alpha-yellow)

## Features

- **QR codes** and **barcodes** (EAN, UPC, Code 128/39/93, PDF417, and more) via zbar
- **Text / OCR** via Tesseract (detects text regions automatically using contour analysis)
- Live camera preview with colour-coded bounding boxes
- **Manual OCR region**: drag a box on the preview to OCR just that area
  (persist until the next click; works while paused)
- Results list with confidence tracking -- **click a result's badge to copy** its value
- Pause/resume scanning (Space bar)
- Automatic confidence decay -- stale results fade and disappear

## Requirements

### System dependencies

These must be installed separately -- they are **not** pip-installable:

| Dependency | Windows                          | macOS / Linux                  |
|-----------|----------------------------------|-------------------------------|
| **zbar**  | `winget install zbar`            | `brew install zbar` / `apt install zbar-tools` |
| **Tesseract** | `winget install tesseract`   | `brew install tesseract` / `apt install tesseract-ocr` |

**Note:** zbar is optional (required only for QR/barcode decoding).  
Tesseract is optional (required only for OCR / text detection).  
The app will work with just OpenCV if neither is installed -- it just won't detect anything.

Both libraries are auto-discovered at startup. If Tesseract isn't on `PATH`,
common install locations (including the winget/UB-Mannheim default under
`%LOCALAPPDATA%\Programs\Tesseract-OCR`) are probed automatically. If zbar's
bundled DLL fails to load, run `readcam install-zbar` to fetch a compatible
build from the latest zbar release.

### Python dependencies

```bash
pip install -r requirements.txt
```

Or, if you prefer an editable install:

```bash
pip install -e .
```

## Usage

```text
readcam [--list-cameras] [--camera INDEX] [--ocr-interval MS] [--frame-interval MS] [--verbose]
```

### List available cameras

```bash
readcam --list-cameras
```

Example output:

```
Available cameras:
  [0] HD Webcam C920
  [1] OBS Virtual Camera
```

> **Note:** Camera names are read from ffmpeg if available. Without ffmpeg, cameras are labelled `Camera 0`, `Camera 1`, etc.

### Scan with a specific camera

```bash
readcam --camera 0
```

### Adjust timing

```bash
# Process OCR every 2 seconds (default 1.5s), capture every 33 ms (~30 fps)
readcam --ocr-interval 2000 --frame-interval 33
```

### Verbose event logging

```bash
# Log scanner events to the console (ISO 8601 timestamps)
readcam --verbose
```

Logged events include: camera discovery, device open/release, zbar
barcode/QR detections, OCR detections (with confidence and timing),
new results entering the list, stale result removal, clipboard copies,
pause/resume, and shutdown.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / resume scanning |
| `Escape` | Quit |
| `R` | Clear all results |
| Drag on preview | Select manual OCR region (persists until next click) |

## User Interface

```
+-----------------------------+---------------------------+
|   Camera Preview            |   Results                 |
|                             |                           |
|   [QR: example.com]  [ ]   |   [QR] example.com        |
|   +------------------+      |       seen 8x, 14s visible|
|   |  QR code here    |      |                           |
|   +------------------+      |   [TXT] Welcome to...     |
|                             |       seen 3x, 5s visible |
|                             |                           |
|   [TXT: Welcome to...]      |                           |
|   +------------------+      |   (click a badge to copy) |
+-----------------------------+---------------------------+
+---------------------------------------------------------+

Colours:
    Green  -- QR codes
    Blue   -- Barcodes
    Orange -- Text (OCR)
```

## Project Structure

```
readcam/
  readcam/
    __init__.py       # Version
    __main__.py       # python -m readcam
    cli.py            # Argument parsing
    config.py         # Defaults and constants
    camera.py         # Camera discovery, CaptureThread, OCRThread
    detector.py       # Text region detection via contours
    results.py        # ResultsManager (dedup, confidence, decay)
    util.py           # Clipboard, time formatting, truncation
    decoders/
      base.py         # Detection dataclass + Decoder ABC
      zbar_decoder.py # QR / barcode via pyzbar
      ocr_decoder.py  # Text via pytesseract
    ui/
      theme.py        # CTk colours and fonts
      app.py          # Main application window
      preview.py      # Camera preview + overlay painting
      results_panel.py # Scrollable results list
```

## Development

```bash
# Editable install
pip install -e .

# Run directly
python -m readcam --list-cameras
python -m readcam --camera 0
```

## Limitations

- **Windows only** (tested). Camera discovery uses DirectShow naming.
- Single camera at a time.
- Tesseract OCR is relatively slow (~1 fps on text-heavy frames).
- Bounding boxes for OCR results may be imprecise (contour-based, not ML).

## License

MIT
