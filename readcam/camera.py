"""Camera discovery and background capture thread."""

import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import sys

import cv2

from .config import OCR_ADAPTIVE_HEADROOM, RESOLUTION_AUTO
from .decoders.ocr_decoder import OCRDecoder
from .decoders.zbar_decoder import ZBarDecoder
from .decoders.zxing_decoder import ZXingDecoder
from .image_enhance import ImageEnhancer

logger = logging.getLogger(__name__)


def _camera_backend() -> int:
    """Preferred OpenCV capture backend for this platform.

    Windows: DirectShow (CAP_DSHOW) opens in well under a second and
    tolerates in-process re-opens; the default MSMF backend is slow
    (tens of seconds) and known to hard-crash on device switches.
    """
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


# Serialises all camera device open/probe/release operations. Windows
# DirectShow drivers can fault when several handles race on the device,
# and concurrent probes must not interleave.
_camera_op_lock = threading.RLock()
import numpy as np

from .config import DEFAULT_FRAME_INTERVAL, RESOLUTION_AUTO
from .image_enhance import ImageEnhancer

from .config import DEFAULT_FRAME_INTERVAL


@dataclass
class CameraInfo:
    """Describes a detected camera device."""
    index: int
    name: str


@dataclass
class ZBarResult:
    """A single result from zbar barcode/QR decoding."""
    type: str            # "QRCODE", "EAN-13", "CODE128", etc.
    data: str            # Decoded text content
    points: list         # Polygon points [(x, y), ...]
    quality: int         # Quality metric from zbar
    rect: Optional[tuple] = None  # (x, y, w, h) derived bounding box


class CameraDiscovery:
    """Probe camera indices to find available devices."""

    MAX_PROBE = 10

    @classmethod
    def quick_indices(cls) -> list[CameraInfo]:
        """Return candidate camera indices without touching any device.

        Instant, lazy alternative to :meth:`list_cameras`: lists indices
        0..MAX_PROBE-1 with generic names so the UI can populate a
        selector immediately. Friendly names are resolved lazily once a
        camera is actually opened.
        """
        return [
            CameraInfo(index=i, name=f"Camera {i}")
            for i in range(cls.MAX_PROBE)
        ]

    @classmethod
    def list_cameras(cls) -> list[CameraInfo]:
        """Return all detected cameras with their names (opens devices).

        Slow (opens each candidate index); the UI only calls this via
        the Refresh button in a background thread. Note failed opens
        emit one OpenCV WARN line on stderr each -- informative, and
        far preferable to the fd-level stderr redirection this module
        used previously (which broke console logging).
        """
        cameras = []
        logger.debug("Probing camera indices 0-%d...", cls.MAX_PROBE - 1)
        for i in range(cls.MAX_PROBE):
            with _camera_op_lock:
                cap = cv2.VideoCapture(i, _camera_backend())
                if cap.isOpened():
                    name = cls._get_camera_name(i)
                    logger.info("Found camera [%d] %s", i, name)
                    cameras.append(CameraInfo(index=i, name=name))
                    cap.release()
        logger.info("Camera discovery complete: %d found", len(cameras))
        return cameras

    @classmethod
    def _get_camera_name(cls, index: int) -> str:
        """Attempt to get a human-readable camera name.

        Tries ffmpeg first (if available), then falls back
        to the generic 'Camera N' label.
        """
        name = cls._try_ffmpeg_name(index)
        if name:
            return name
        return f"Camera {index}"

    @classmethod
    def _try_ffmpeg_name(cls, index: int) -> Optional[str]:
        """Query ffmpeg DirectShow devices for a friendly name."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True, text=True, timeout=10,
            )
            # ffmpeg lists DirectShow devices in stderr like:
            #   "HD Webcam C920" (usb, ...
            # Parse by scanning for device entries
            lines = result.stderr.splitlines()
            in_device_list = False
            device_index = 0
            for line in lines:
                stripped = line.strip()
                if "DirectShow device list" in stripped:
                    in_device_list = True
                    continue
                if in_device_list and ("Alternate name" in stripped or not stripped):
                    continue
                if in_device_list and '"' in stripped:
                    if device_index == index:
                        # Extract name between quotes
                        start = stripped.find('"')
                        end = stripped.find('"', start + 1)
                        if start != -1 and end != -1:
                            return stripped[start + 1:end]
                    device_index += 1
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None


class CaptureThread(threading.Thread):
    """Dedicated thread for camera capture and fast zbar decoding.

    Captures frames from the selected camera, runs zbar decode on
    each frame, and pushes the frame + results to a shared queue
    for the UI thread to consume.
    """

    def __init__(
        self,
        camera_index: int,
        frame_queue: queue.Queue,
        ocr_queue: queue.Queue,
        pause_event: threading.Event,
        stop_event: threading.Event,
        frame_interval_ms: int = DEFAULT_FRAME_INTERVAL,
        ocr_interval_ms: int = 1500,
        ocr_interval_holder: Optional[dict] = None,
        auto_holder: Optional[dict] = None,
        camera_index_holder: Optional[dict] = None,
        transform_holder: Optional[dict] = None,
        filter_holder: Optional[dict] = None,
        debug: bool = False,
    ):
        super().__init__(name="capture", daemon=True)
        self.camera_index = camera_index
        # Shared desired camera index; the thread re-opens when it changes
        self.camera_index_holder = (
            camera_index_holder if camera_index_holder is not None
            else {"index": camera_index}
        )
        self.frame_queue = frame_queue
        self.ocr_queue = ocr_queue
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.frame_interval = frame_interval_ms / 1000.0
        self.ocr_interval = ocr_interval_ms / 1000.0
        # Shared with the OCR thread, which raises the interval adaptively
        self.ocr_interval_holder = (
            ocr_interval_holder if ocr_interval_holder is not None
            else {"ms": ocr_interval_ms}
        )
        # Shared auto-scan toggle; False disables zbar + OCR frame pushes
        self.auto_holder = auto_holder if auto_holder is not None else {"enabled": True}
        # Shared display transform: rotation in degrees (0/90/180/270) and
        # flip mode ("none"/"h"/"v"/"hv"). Applied to every frame before
        # decoding so all boxes and the ROI selection stay consistent.
        self.transform_holder = (
            transform_holder if transform_holder is not None
            else {"rotation": 0, "flip": "none"}
        )
        # Shared live filter settings: colour mode + image adjustments
        # applied here, plus the resolution requested from the panel.
        self.filter_holder = (
            filter_holder if filter_holder is not None else {}
        )
        self._enhancer = ImageEnhancer(self.filter_holder)
        self._current_resolution: str = self.filter_holder.get(
            "resolution", RESOLUTION_AUTO,
        )
        self.debug = debug

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_ocr_time = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()
        self.current_fps = 0.0
        self._camera_name: Optional[str] = None
        self._open_failed_logged = False
        self._read_failures = 0
        self._open_attempts = 0

        # UI-visible state (plain attributes; GIL-safe reads)
        self.connected = False
        self.open_failed = False
        self.gave_up = False

    def run(self) -> None:
        """Main capture loop."""
        while not self.stop_event.is_set():
            # Live camera switching: re-open when the desired index changes
            desired = self.camera_index_holder.get("index")
            if desired != self.camera_index:
                logger.info(
                    "Camera switch requested: %d -> %d",
                    self.camera_index, desired,
                )
                self._release_camera()
                self.camera_index = desired
                self._camera_name = None  # re-resolve on next access
                self._open_failed_logged = False
                self._open_attempts = 0
                self.open_failed = False
                self.gave_up = False
                self.connected = False
                # Drain queues so stale frames from the previous
                # camera never render or get OCR'd.
                self._drain_queue(self.frame_queue)
                self._drain_queue(self.ocr_queue)

            if self._cap is None:
                # Try a few times, then give up until the selection
                # changes -- retrying forever against a dead index just
                # hammers the device stack and spams the log.
                if self._open_attempts < 3:
                    self._open_attempts += 1
                    if self._open_camera(self.camera_index):
                        self._apply_resolution(self._current_resolution)
                        self.connected = True
                        self.open_failed = False
                        self.gave_up = False
                        self._open_failed_logged = False
                        self._fps_counter = 0
                        self._fps_timer = time.time()
                    else:
                        self.open_failed = True
                        if not self._open_failed_logged:
                            logger.error(
                                "Failed to open camera %d; %d more attempt(s) "
                                "before giving up",
                                self.camera_index, 3 - self._open_attempts,
                            )
                            self._open_failed_logged = True
                        time.sleep(1.0)
                    continue
                # Given up: idle cheaply until a new selection arrives
                self.gave_up = True
                time.sleep(0.2)
                continue

            if self.pause_event.is_set():
                time.sleep(0.05)
                continue

            # Live resolution switching: re-open when the requested
            # resolution changes. Many CAP_DSHOW cameras only honour
            # frame-size properties on open, so a re-open is the
            # reliable way to apply it.
            desired_resolution = self.filter_holder.get(
                "resolution", RESOLUTION_AUTO,
            )
            if desired_resolution != self._current_resolution:
                logger.info(
                    "Resolution switch requested: %s -> %s",
                    self._current_resolution, desired_resolution,
                )
                self._release_camera()
                self._current_resolution = desired_resolution
                self._open_attempts = 0
                self._open_failed_logged = False
                self.open_failed = False
                self.gave_up = False
                self.connected = False
                self._drain_queue(self.frame_queue)
                self._drain_queue(self.ocr_queue)
                continue

            try:
                ret, frame = self._cap.read()
            except Exception:
                logger.exception("Frame read raised; re-opening camera")
                self._release_camera()
                continue
            if not ret:
                self._read_failures += 1
                logger.warning(
                    "Frame read failed (%d); retrying", self._read_failures,
                )
                if self._read_failures >= 10:
                    logger.error(
                        "Camera %d unreadable; re-opening", self.camera_index,
                    )
                    self._release_camera()
                    self._read_failures = 0
                time.sleep(0.05)
                continue
            self._read_failures = 0

            frame = self._apply_transform(frame)

            # Keep a copy of the raw frame for the OCR pipeline: the
            # user's colour-mode / adjustments are display choices and
            # should not degrade Tesseract quality (OCR does its own
            # preprocessing). Same dimensions, so ROI boxes stay valid.
            ocr_frame = frame.copy()

            # Apply colour mode + image adjustments for display and
            # zbar decoding (1-bit B/W generally helps zbar detection).
            frame = self._enhancer.apply(frame)

            try:
                # Fast decode: run zbar on the frame (auto mode only)
                zbar_results = []
                if self.auto_holder.get("enabled"):
                    zbar_results = self._decode_zbar(frame)
                    if zbar_results:
                        for zr in zbar_results:
                            logger.info(
                                "zbar: %s %r (quality=%d)",
                                zr.type, zr.data, zr.quality,
                            )
                else:
                    logger.debug("Auto-scan disabled; skipping zbar decode")

                # Push to UI queue (non-blocking, drop if lagging)
                try:
                    self.frame_queue.put_nowait((frame, zbar_results))
                except queue.Full:
                    logger.debug("frame_queue full; dropped frame")

                # Periodically push a frame to the OCR pipeline (auto mode)
                now = time.time()
                if self.auto_holder.get("enabled") and now - self._last_ocr_time >= self.ocr_interval:
                    self._last_ocr_time = now
                    # Pick up adaptive interval updates from the OCR thread
                    adaptive_ms = self.ocr_interval_holder.get("ms")
                    if adaptive_ms and abs(adaptive_ms / 1000.0 - self.ocr_interval) > 0.001:
                        self.ocr_interval = adaptive_ms / 1000.0
                        logger.info(
                            "Adaptive OCR interval updated to %.0f ms",
                            adaptive_ms,
                        )
                    try:
                        self.ocr_queue.put_nowait(ocr_frame)
                        logger.debug("Pushed frame to OCR queue")
                    except queue.Full:
                        logger.debug("ocr_queue full; dropped OCR frame")

                # Track FPS
                self._fps_counter += 1
                elapsed = time.time() - self._fps_timer
                if elapsed >= 1.0:
                    self.current_fps = self._fps_counter / elapsed
                    logger.debug("Capture FPS: %.1f", self.current_fps)
                    self._fps_counter = 0
                    self._fps_timer = time.time()

                # Throttle
                time.sleep(self.frame_interval)
            except Exception:
                logger.exception(
                    "Capture loop error; releasing camera %d and retrying",
                    self.camera_index,
                )
                self._release_camera()
                self._open_attempts = 0
                self.connected = False
                time.sleep(0.5)

        self._release_camera()
        logger.info("Camera %d released; capture thread exiting", self.camera_index)

    def _open_camera(self, index: int) -> bool:
        """Open the camera at *index*; returns True on success."""
        logger.info("Opening camera %d...", index)
        with _camera_op_lock:
            self._cap = cv2.VideoCapture(index, _camera_backend())
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                return False

            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera %d opened (%dx%d)",
            index, width, height,
        )
        # Resolve the friendly name once per open (may shell out to ffmpeg)
        self._camera_name = CameraDiscovery._get_camera_name(index)
        logger.info("Camera %d name: %s", index, self._camera_name)
        return True

    def _release_camera(self) -> None:
        """Release the current camera, if any."""
        with _camera_op_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    logger.debug("Error releasing camera", exc_info=True)
                self._cap = None

    def _apply_resolution(self, resolution: str) -> None:
        """Request the requested frame size from the open camera.

        "Auto" leaves the camera at its default. Applied on every open
        and whenever the panel resolution changes (via a re-open).
        """
        if resolution == RESOLUTION_AUTO or self._cap is None:
            return
        try:
            width, height = (int(v) for v in resolution.split("x", 1))
        except ValueError:
            logger.warning("Invalid resolution %r; ignoring", resolution)
            return
        with _camera_op_lock:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual != (width, height):
            logger.warning(
                "Camera %d rejected %s; reporting %dx%d",
                self.camera_index, resolution, actual[0], actual[1],
            )
        else:
            logger.info(
                "Camera %d resolution set to %s",
                self.camera_index, resolution,
            )

    @staticmethod
    def _drain_queue(q: queue.Queue) -> None:
        """Empty a queue without blocking (drop stale items)."""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _apply_transform(self, frame: np.ndarray) -> np.ndarray:
        """Apply the user's rotation / flip settings to *frame*.

        Reads the shared transform holder on each frame (GIL-safe dict
        reads) so the UI can change orientation live.
        """
        rotation = self.transform_holder.get("rotation", 0)
        flip = self.transform_holder.get("flip", "none")
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if flip == "h":
            frame = cv2.flip(frame, 1)
        elif flip == "v":
            frame = cv2.flip(frame, 0)
        elif flip == "hv":
            frame = cv2.flip(frame, -1)
        return frame

    def _decode_zbar(self, frame: np.ndarray) -> list[ZBarResult]:
        """Run zbar barcode/QR decoding on the frame."""
        try:
            from pyzbar.pyzbar import decode
        except (ImportError, OSError):
            logger.warning("pyzbar unavailable; barcode/QR decoding disabled")
            return []

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            symbols = decode(gray)
            results = []
            for sym in symbols:
                data = sym.data
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="replace")
                else:
                    data = str(data)
                # pyzbar exposes the symbol location as `polygon` (a list
                # of Point namedtuples); there is no `points` attribute.
                points = [
                    (int(p.x), int(p.y))
                    for p in (getattr(sym, 'polygon', None) or [])
                ]
                sym_type = getattr(sym, 'type', 'UNKNOWN')
                # pyzbar returns a ZBarSymbol IntEnum; str() of it is the
                # numeric value on Python 3.11+, so use .name and map it
                # through the decoder's canonical type table.
                sym_name = str(getattr(sym_type, 'name', sym_type)).upper()
                results.append(ZBarResult(
                    type=ZBarDecoder.TYPE_MAP.get(sym_name, sym_name),
                    data=data,
                    points=points,
                    quality=getattr(sym, 'quality', 0),
                    rect=self._points_to_rect(points) or self._rect_from_sym(sym),
                ))
            return results
        except Exception as exc:
            logger.debug("zbar decode error: %s", exc)
            return []

    @staticmethod
    def _points_to_rect(points: list) -> Optional[tuple]:
        """Convert zbar polygon points to a (x, y, w, h) rect."""
        if not points:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    @staticmethod
    def _rect_from_sym(sym) -> Optional[tuple]:
        """Fall back to pyzbar's Rect when no polygon is available."""
        rect = getattr(sym, 'rect', None)
        if rect is None:
            return None
        return (rect.left, rect.top, rect.width, rect.height)

    @property
    def camera_name(self) -> str:
        """Return the human-readable camera name."""
        if self._camera_name is None:
            cameras = CameraDiscovery.list_cameras()
            for cam in cameras:
                if cam.index == self.camera_index:
                    self._camera_name = cam.name
                    break
            if self._camera_name is None:
                self._camera_name = f"Camera {self.camera_index}"
        return self._camera_name

    @property
    def current_index_name(self) -> str:
        """Human-readable name of the active camera (cheap; no probing)."""
        return self._camera_name or f"Camera {self.camera_index}"


class OCRThread(threading.Thread):
    """Background thread for optical character recognition.

    Consumes frames from the OCR queue, runs whole-frame Tesseract
    OCR, and pushes the resulting text detections to the result queue.
    """

    def __init__(
        self,
        ocr_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
        pause_event: threading.Event,
        roi_holder: Optional[dict] = None,
        ocr_interval_holder: Optional[dict] = None,
        auto_holder: Optional[dict] = None,
        filter_holder: Optional[dict] = None,
        debug: bool = False,
    ):
        super().__init__(name="ocr", daemon=True)
        self.ocr_queue = ocr_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        # Shared ROI (x, y, w, h) set by UI mouse selection, or None
        self.roi_holder = roi_holder if roi_holder is not None else {"roi": None}
        # Shared auto-scan toggle; False drops whole-frame OCR jobs
        self.auto_holder = auto_holder if auto_holder is not None else {"enabled": True}
        # Shared live filter thresholds written by the UI sliders
        self.filter_holder = (
            filter_holder if filter_holder is not None else {}
        )
        # Shared with the capture thread, which applies our adaptive interval
        self.ocr_interval_holder = (
            ocr_interval_holder if ocr_interval_holder is not None
            else {"ms": 1500}
        )
        self.debug = debug

        self._decoder = None

    def run(self) -> None:
        """Main OCR processing loop."""
        logger.info("OCR thread starting...")
        self._decoder = OCRDecoder(filter_holder=self.filter_holder, debug=self.debug)
        self._zxing = ZXingDecoder(debug=self.debug)
        self._zbar = ZBarDecoder(debug=self.debug)

        logger.info(
            "OCR thread decoders: Tesseract=%s, zxing=%s, zbar=%s",
            self._decoder.available, self._zxing.available, self._zbar.available,
        )

        if not self._decoder.available:
            logger.warning("Tesseract unavailable; text OCR disabled")
            # Keep running: barcode decoders still operate in this thread
            if not (self._zxing.available or self._zbar.available):
                return
        else:
            logger.info("OCR thread ready (Tesseract detected)")

        while not self.stop_event.is_set():
            # No pause check here: while paused the queue simply stays
            # empty, and manual ROI selections should still be processed.

            try:
                frame = self.ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Crop to the manual selection region if one is active
            roi = self.roi_holder.get("roi")
            offset = (0, 0)
            if roi is None and not self.auto_holder.get("enabled"):
                # Manual-only mode: ignore whole-frame jobs with no selection
                logger.debug("Manual-only mode and no ROI; dropping OCR job")
                continue
            if roi is not None:
                rx, ry, rw, rh = roi
                fh, fw = frame.shape[:2]
                x2, y2 = min(rx + rw, fw), min(ry + rh, fh)
                x1, y1 = max(rx, 0), max(ry, 0)
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                frame = frame[y1:y2, x1:x2]
                offset = (x1, y1)
                logger.debug("OCR using manual ROI %s", roi)

            start = time.perf_counter()
            detections: list = []
            try:
                if self._decoder.available:
                    detections = self._decoder.decode(frame)
            except Exception as exc:
                logger.debug("OCR processing error: %s", exc)

            # Barcode/QR/DataMatrix decode also runs in this thread:
            # full frame in auto mode, cropped region in manual mode.
            # zxing-cpp picks up Data Matrix codes that zbar misses.
            for barcode_decoder in (self._zxing, self._zbar):
                if not barcode_decoder.available:
                    continue
                try:
                    detections.extend(barcode_decoder.decode(frame))
                except Exception as exc:
                    logger.debug(
                        "%s decode error: %s", type(barcode_decoder).__name__, exc,
                    )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Adaptive throttle: if the pass took longer than the current
            # interval, raise the interval so capture FPS stays healthy.
            current_ms = self.ocr_interval_holder.get("ms", 1500)
            if elapsed_ms * OCR_ADAPTIVE_HEADROOM > current_ms:
                adaptive_ms = int(elapsed_ms * OCR_ADAPTIVE_HEADROOM)
                self.ocr_interval_holder["ms"] = adaptive_ms
                logger.info(
                    "OCR pass took %.0f ms (> %.0f ms interval); "
                    "raising OCR interval to %d ms",
                    elapsed_ms, current_ms, adaptive_ms,
                )

            if detections:
                # Shift ROI-relative boxes back into frame coordinates
                if offset != (0, 0):
                    ox, oy = offset
                    for det in detections:
                        if det.bbox:
                            bx, by, bw, bh = det.bbox
                            det.bbox = (bx + ox, by + oy, bw, bh)
                for det in detections:
                    logger.info(
                        "ocr: %s %r (conf=%.2f, bbox=%s) [%.0f ms]",
                        det.type, det.value, det.confidence,
                        det.bbox, elapsed_ms,
                    )
                try:
                    self.result_queue.put(detections)
                except Exception:
                    pass