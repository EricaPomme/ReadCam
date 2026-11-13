# ReadCam

Live camera scanner that detects QR codes, barcodes, and text in real time.

## Style

- Follow the style guide in `~/.config/opencode/AGENTS.md` (global defaults).
- Canadian English spelling, metric units, ISO 8601 dates.
- ASCII-only source files.
- Python: type hints, `pathlib` for paths, `main() -> int`, `if __name__ == "__main__"`.

## Architecture

Single-package layout under `readcam/`. Three-thread design:

| Thread | Role |
|--------|------|
| **Capture** | Camera I/O + fast zbar decode (every frame). Pushes to `frame_queue`. |
| **OCR** | Contour detection + Tesseract (every ~1.5 s). Pushes to `ocr_result_queue`. |
| **UI (main)** | Polls queues, renders preview + results. CustomTkinter. |

### Queues

- `frame_queue: Queue(maxsize=2)` -- latest frame + zbar results. Capture drops frames if UI lags.
- `ocr_queue: Queue(maxsize=2)` -- frame copies for OCR thread. Drops if OCR busy.
- `ocr_result_queue: Queue()` -- text detections from OCR thread.

### Thread Control

- `pause_event: Event` -- set when paused; threads skip processing but hold their state.
- `stop_event: Event` -- set during shutdown; threads exit their main loop.

## Key Design Decisions

### UI Framework: CustomTkinter
Dark mode, green accent. Bounding boxes drawn with OpenCV on the frame before PIL/CTkImage conversion.

### Scanned Results Lifecycle
- Each unique `(type, value)` pair tracked once by `ResultsManager`.
- **Confidence** increments on each re-detection (capped at 100).
- **Decay**: confidence drops by ~1 per 2 seconds once unseen for 5 s.
- **Removal**: purged entirely after 30 s of no sightings.
- **Text threshold**: OCR text requires 3 detections before it appears (reduces false positives).
- **Live filter sliders**: a Filters panel below the preview (always visible) adjusts the text thresholds and decay/removal delays at runtime via a shared `filter_holder` dict; consumers fall back to config defaults for missing keys.

### Bounding Box Colours
- Green: QR codes
- Blue: barcodes (EAN, UPC, Code 128/39/93, PDF417, etc.)
- Orange: text (OCR)
- Grey: unknown/unrecognised

### Capture Saving
- **Capture panel** below the preview (row 2): clickable output path label (defaults to `~`, doubles as the directory chooser), Disk (save to disk), Clip (copy to clipboard), and a BB checkbox (include bounding boxes in the output).
- Disk saves `ReadCam_YYYY-MM-DD_HH-MM-SS.png` (`_2`-style suffix on same-second collision); Clip uses a CF_DIB via the Win32 clipboard API and is Windows-only (disabled elsewhere).
- Overlay drawing is shared: `preview.draw_overlays()` renders both the live preview and the BB-checked capture.

### Decoder Availability
Both `pyzbar` (QR/barcode) and `pytesseract` (OCR) are optional. The app detects their absence at startup and shows status in the status bar. No crashes if missing.

### Camera Discovery
Startup is lazy and instant: the dropdown lists candidate indices 0-9 (`Camera N`) without touching any device. Friendly names are resolved in the capture thread once a camera opens (via `ffmpeg -list_devices`, fallback `Camera N`) and folded into the dropdown by the UI. The **Refresh** button runs a full probe (`CameraDiscovery.list_cameras`, opens each device) in a background thread.

Camera switching is live: the UI writes to `camera_index_holder` and `CaptureThread` re-opens the device and drains stale frames. A new index gets 3 open attempts; on failure the thread gives up (idle until the next selection) and the status bar reports "Camera N unavailable". On Windows the capture backend is `CAP_DSHOW` (fast opens, stable re-opens); all device open/probe/release operations are serialised by `_camera_op_lock`.

### Manual OCR Region (ROI)
Dragging on the preview sets a selection (frame coordinates) exposed via
`PreviewPanel.selection`. The app syncs it into `roi_holder` each tick;
the OCR thread crops frames to the ROI before decoding and offsets result
boxes back into frame coordinates. Mouse-up triggers one immediate OCR
pass. The OCR thread deliberately ignores the pause event: with capture
paused the queue stays empty anyway, and manual selections must still OCR.

## CLI

```text
readcam [--list-cameras] [--camera INDEX] [--ocr-interval MS]
        [--frame-interval MS] [--rotate DEG] [--flip none|h|v|hv]
        [--debug]
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `Escape` | Quit |
| `R` | Clear all results |

## Dependencies

### Python packages (pip)
`opencv-python`, `customtkinter`, `pillow`, `pyzbar`, `pytesseract`, `pyperclip`

### System (install separately per platform)
- **zbar** -- `winget install zbar` / `brew install zbar` / `apt install zbar-tools`
- **Tesseract** -- `winget install tesseract` / `brew install tesseract` / `apt install tesseract-ocr`

## Versioning

- Version tracked in `readcam/__init__.py` (`__version__`).
- Semantic versioning: `major.minor.revision`.
- Bump only at release close-out; never bump on individual commits.

## Commit Rules

- Conventional Commits format (`feat:`, `fix:`, `chore:`, etc.).
- `## Recent Changes` section in `CHANGELOG.md` updated alongside code changes.
- `release` skill flattens Recent Changes into a version section.
- No `Co-authored-by:` or other trailers.