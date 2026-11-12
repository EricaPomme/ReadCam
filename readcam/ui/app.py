"""Main ReadCam application window built with CustomTkinter."""

import queue
import threading
import time
from typing import Optional

import customtkinter as ctk

from ..camera import CaptureThread, CameraDiscovery, OCRThread
from ..config import (
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_OCR_INTERVAL,
    UI_MIN_HEIGHT,
    UI_MIN_WIDTH,
    UI_PREVIEW_MIN_WIDTH,
    UI_RESULTS_MIN_WIDTH,
    UI_TITLE,
)
from ..decoders.ocr_decoder import OCRDecoder
from ..decoders.zbar_decoder import ZBarDecoder
from ..results import ResultsManager
from .preview import PreviewPanel
from .results_panel import ResultsPanel
from .theme import FONTS, setup_theme


class ReadCamApp(ctk.CTk):
    """Main ReadCam application window."""

    def __init__(
        self,
        camera_index: int = 0,
        ocr_interval_ms: int = DEFAULT_OCR_INTERVAL,
        frame_interval_ms: int = DEFAULT_FRAME_INTERVAL,
        debug: bool = False,
    ):
        setup_theme()
        super().__init__()

        self.title(UI_TITLE)
        self.minsize(UI_MIN_WIDTH, UI_MIN_HEIGHT)

        # Grid: preview (left) | results (right)
        self.grid_columnconfigure(0, weight=1, minsize=UI_PREVIEW_MIN_WIDTH)
        self.grid_columnconfigure(1, weight=0, minsize=UI_RESULTS_MIN_WIDTH)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)  # status bar

        # Configuration
        self.camera_index = camera_index
        self.ocr_interval_ms = ocr_interval_ms
        self.frame_interval_ms = frame_interval_ms
        self.debug = debug

        # Thread communication
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.ocr_queue: queue.Queue = queue.Queue(maxsize=2)
        self.ocr_result_queue: queue.Queue = queue.Queue()

        # Thread control
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self._paused = False

        # State
        self._latest_frame: Optional = None
        self._latest_zbar: list = []
        self._camera_connected = False

        # Decoder availability (checked once at startup)
        self._ocr_available = OCRDecoder._check_available()
        self._zbar_available = ZBarDecoder._check_available()

        # Results
        self.results_manager = ResultsManager()

        # --- Build UI ---
        self.preview_panel = PreviewPanel(self)
        self.preview_panel.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.results_panel = ResultsPanel(self)
        self.results_panel.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.status_bar = ctk.CTkLabel(
            self,
            text="Initialising...",
            font=FONTS["small"],
            text_color="#888888",
            anchor="w",
        )
        self.status_bar.grid(
            row=1, column=0, columnspan=2,
            padx=5, pady=(0, 2),
            sticky="ew",
        )

        # --- Start background threads ---
        self.capture_thread = CaptureThread(
            camera_index=camera_index,
            frame_queue=self.frame_queue,
            ocr_queue=self.ocr_queue,
            pause_event=self.pause_event,
            stop_event=self.stop_event,
            frame_interval_ms=frame_interval_ms,
            ocr_interval_ms=ocr_interval_ms,
            debug=debug,
        )
        self.capture_thread.start()

        self.ocr_thread = OCRThread(
            ocr_queue=self.ocr_queue,
            result_queue=self.ocr_result_queue,
            stop_event=self.stop_event,
            pause_event=self.pause_event,
            debug=debug,
        )
        self.ocr_thread.start()

        # --- Periodic UI update ---
        self._ui_fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self.after(50, self._update_loop)

        # --- Clean shutdown ---
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Keyboard shortcuts ---
        self.bind("<space>", lambda _e: self._toggle_pause())
        self.bind("<Escape>", lambda _e: self._on_close())
        self.bind("<Key-r>", lambda _e: self.results_manager.clear())

    # ------------------------------------------------------------------
    # UI update loop (runs on the main thread via .after())
    # ------------------------------------------------------------------

    def _update_loop(self) -> None:
        """Periodic update: poll queues, redraw preview, refresh results."""
        if not self._paused:
            # 1. Consume latest frame from capture thread
            try:
                while True:
                    self._latest_frame, self._latest_zbar = (
                        self.frame_queue.get_nowait()
                    )
                    self._camera_connected = True
            except queue.Empty:
                pass

            # 2. Consume OCR results
            try:
                while True:
                    text_detections = self.ocr_result_queue.get_nowait()
                    for det in text_detections:
                        self.results_manager.add_text_hit(det.value, det.bbox)
            except queue.Empty:
                pass

            # 3. Add zbar results
            for zr in self._latest_zbar:
                self.results_manager.add_or_update(zr.type, zr.data, zr.rect)

            # 4. Decay stale results
            self.results_manager.decay()

            # 5. Redraw preview
            self.preview_panel.update(
                self._latest_frame,
                self._latest_zbar,
                self.results_manager.get_active(),
            )

            # 6. Refresh results list
            self.results_panel.update_results(
                self.results_manager.get_sorted(),
            )

            # 7. Track FPS for status display
            self._frame_count += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self._ui_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.time()

        # --- Update status bar ---
        parts: list[str] = []

        # Camera status
        if self._camera_connected:
            parts.append(f"Camera {self.camera_index}")
        else:
            parts.append("Connecting...")

        if self._ui_fps > 0:
            parts.append(f"{self._ui_fps:.0f} FPS")

        rc = self.results_manager.count
        parts.append(f"{rc} result{'s' if rc != 1 else ''}")

        if self._paused:
            parts.append("PAUSED")

        if not self._zbar_available:
            parts.append("zbar unavailable")
        if not self._ocr_available:
            parts.append("OCR unavailable")

        self.status_bar.configure(
            text=" \u00b7 ".join(parts) if parts else "No camera feed",
        )

        self.after(50, self._update_loop)

    # ------------------------------------------------------------------
    # App lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Enter the CTk main loop."""
        self.mainloop()

    def _toggle_pause(self) -> None:
        """Pause or resume scanning."""
        self._paused = not self._paused
        if self._paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()

    def _on_close(self) -> None:
        """Clean up threads and close the window."""
        self.stop_event.set()
        try:
            self.destroy()
        except Exception:
            pass