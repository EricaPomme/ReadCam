# ReadCam

Live camera scanner that detects QR codes, barcodes, and text in real time.

## Features

- **QR codes** and **barcodes** (EAN, UPC, Code 128/39/93, PDF417, and more) via zbar
- **Text / OCR** via Tesseract (detects text regions automatically using contour analysis)
- Live camera preview with colour-coded bounding boxes
- **Manual OCR region**: drag a box on the preview to OCR just that area
  (persist until the next click; works while paused)
- Results list with confidence tracking -- **click a result's badge to copy** its value
- **Capture saving**: save the current frame to disk or copy it to the
  clipboard as a PNG; optionally include bounding boxes and labels
- Pause/resume scanning (Space bar)
- Automatic confidence decay -- stale results fade and disappear

## Requirements

### System dependencies

These must be installed separately -- they are **not** pip-installable:

| Dependency | Windows                          | macOS / Linux                  |
|-----------|----------------------------------|-------------------------------|
| **zbar**  | `winget install zbar`            | `brew install zbar` / `apt install zbar-tools` |
| **Tesseract** | `winget install tesseract`   | `brew install tesseract` / `apt install tesseract-ocr` |

**Note:** zbar is optional (QR/barcode decoding; Data Matrix is handled
by `zxing-cpp`, which installs via pip with no system dependencies).  
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
readcam [--list-cameras] [--camera INDEX] [--ocr-interval MS] [--frame-interval MS]
        [--rotate DEG] [--flip none|h|v|hv] [--verbose]
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

### Rotate / flip the camera view

```bash
# Rotate the view 90 degrees clockwise and mirror horizontally
readcam --rotate 90 --flip h
```

`--rotate` accepts 0, 90, 180, or 270 (degrees, clockwise). `--flip`
accepts `none`, `h` (horizontal), `v` (vertical), or `hv` (both). The
orientation can also be changed at any time while running with the
**Rotate** and **Flip** buttons in the toolbar; the transform applies
before decoding, so bounding boxes, OCR results, and the manual ROI
selection stay consistent.

### Verbose event logging

```bash
# Log scanner events to the console (ISO 8601 timestamps)
readcam --verbose
```

### Manual-only mode

```bash
# No automatic scanning -- only the drag-selected region is decoded
readcam --no-auto
```

In this mode the camera feed displays normally, but nothing is scanned
until you drag a selection box on the preview. The status bar shows a
`MANUAL` button (yellow) instead of `AUTO` (blue) -- both are live
toggle buttons, so the mode can be switched at any time while running.
Dragging still triggers an immediate OCR pass on mouse-up.

### Capture saving

The panel below the preview saves or copies the current frame:

- **Output path**: the directory captures are written to (defaults to
  `~`). Clicking the path opens a directory chooser.
- **Disk**: saves the current frame as
  `ReadCam_YYYY-MM-DD_HH-MM-SS.png` in the output directory (a
  `_2`-style suffix is appended on a same-second collision).
- **Clip**: copies the current frame image to the clipboard.
- **BB**: when checked, bounding boxes and labels are included in the
  captured output; unchecked, the raw frame is captured.

> **Note:** Clipboard image capture (**Clip**) is only implemented for
> Windows (CF_DIB). On non-Windows platforms the button is disabled --
> not yet implemented.

Logged events include: camera discovery, device open/release, zbar
barcode/QR detections, OCR detections (with confidence and timing),
new results entering the list, stale result removal, clipboard copies,
capture saves, pause/resume, and shutdown.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / resume scanning |
| `A` | Toggle auto-scan / manual-only |
| `Escape` | Quit |
| `R` | Clear all results |
| Drag on preview | Select manual OCR region (persists until next click) |

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
