"""Camera discovery and background capture thread."""

import logging
import os
import queue
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import cv2

from .config import OCR_ADAPTIVE_HEADROOM
from .decoders.ocr_decoder import OCRDecoder

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_opencv_stderr():
    """Temporarily redirect stderr to suppress OpenCV C++ noise."""
    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
import numpy as np

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
    def list_cameras(cls) -> list[CameraInfo]:
        """Return all detected cameras with their names."""
        cameras = []
        logger.debug("Probing camera indices 0-%d...", cls.MAX_PROBE - 1)
        with _suppress_opencv_stderr():
            for i in range(cls.MAX_PROBE):
                cap = cv2.VideoCapture(i, cv2.CAP_ANY)
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
        debug: bool = False,
    ):
        super().__init__(name="capture", daemon=True)
        self.camera_index = camera_index
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
        self.debug = debug

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_ocr_time = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()
        self.current_fps = 0.0
        self._camera_name: Optional[str] = None

    def run(self) -> None:
        """Main capture loop."""
        logger.info("Opening camera %d...", self.camera_index)
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error("Failed to open camera %d", self.camera_index)
            return

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera %d opened (%dx%d); capture loop started "
            "(frame interval %.0f ms, OCR interval %.0f ms)",
            self.camera_index, width, height,
            self.frame_interval * 1000, self.ocr_interval * 1000,
        )

        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame read failed; retrying")
                time.sleep(0.05)
                continue

            # Fast decode: run zbar on the frame
            zbar_results = self._decode_zbar(frame)
            if zbar_results:
                for zr in zbar_results:
                    logger.info(
                        "zbar: %s %r (quality=%d)",
                        zr.type, zr.data, zr.quality,
                    )

            # Push to UI queue (non-blocking, drop if lagging)
            try:
                self.frame_queue.put_nowait((frame, zbar_results))
            except queue.Full:
                logger.debug("frame_queue full; dropped frame")

            # Periodically push a frame to the OCR pipeline
            now = time.time()
            if now - self._last_ocr_time >= self.ocr_interval:
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
                    self.ocr_queue.put_nowait(frame.copy())
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

        self._cap.release()
        logger.info("Camera %d released; capture thread exiting", self.camera_index)

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
                points = getattr(sym, 'points', [])
                results.append(ZBarResult(
                    type=str(getattr(sym, 'type', 'UNKNOWN')).upper(),
                    data=data,
                    points=points,
                    quality=getattr(sym, 'quality', 0),
                    rect=self._points_to_rect(points),
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
        ocr_interval_holder: Optional[dict] = None,
        debug: bool = False,
    ):
        super().__init__(name="ocr", daemon=True)
        self.ocr_queue = ocr_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
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
        self._decoder = OCRDecoder(debug=self.debug)

        if not self._decoder.available:
            logger.warning("Tesseract unavailable; OCR disabled")
            return

        logger.info("OCR thread ready (Tesseract detected)")

        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue

            try:
                frame = self.ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            start = time.perf_counter()
            try:
                detections = self._decoder.decode(frame)
            except Exception as exc:
                logger.debug("OCR processing error: %s", exc)
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Adaptive throttle: if OCR took longer than the current
            # interval, raise the interval so capture FPS stays healthy.
            current_ms = self.ocr_interval_holder.get("ms", 1500)
            if elapsed_ms * OCR_ADAPTIVE_HEADROOM > current_ms:
                adaptive_ms = int(elapsed_ms * OCR_ADAPTIVE_HEADROOM)
                self.ocr_interval_holder["ms"] = adaptive_ms
                logger.info(
                    "OCR took %.0f ms (> %.0f ms interval); "
                    "raising OCR interval to %d ms",
                    elapsed_ms, current_ms, adaptive_ms,
                )

            if detections:
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