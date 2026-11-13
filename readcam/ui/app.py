"""Main ReadCam application window built with CustomTkinter."""

import logging
import queue
import threading
import time
from typing import Optional

import customtkinter as ctk

logger = logging.getLogger(__name__)

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
from .theme import COLOURS, FONTS, setup_theme

logger = logging.getLogger(__name__)


class ReadCamApp(ctk.CTk):
    """Main ReadCam application window."""

    def __init__(
        self,
        camera_index: int = 0,
        ocr_interval_ms: int = DEFAULT_OCR_INTERVAL,
        frame_interval_ms: int = DEFAULT_FRAME_INTERVAL,
        no_auto: bool = False,
        debug: bool = False,
    ):
        setup_theme()
        super().__init__()

        self.title(UI_TITLE)
        self.minsize(UI_MIN_WIDTH, UI_MIN_HEIGHT)

        # Grid: preview (left, fixed-ish) | results (right, expands)
        self.grid_columnconfigure(0, weight=0, minsize=UI_PREVIEW_MIN_WIDTH)
        self.grid_columnconfigure(1, weight=1, minsize=UI_RESULTS_MIN_WIDTH)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)  # status bar

        # Configuration
        self.camera_index = camera_index
        self.ocr_interval_ms = ocr_interval_ms
        self.frame_interval_ms = frame_interval_ms
        self.no_auto = no_auto
        self.debug = debug

        # Thread communication
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.ocr_queue: queue.Queue = queue.Queue(maxsize=2)
        self.ocr_result_queue: queue.Queue = queue.Queue()
        # Shared, mutable OCR interval (ms) -- OCR thread raises it adaptively
        self.ocr_interval_holder = {"ms": ocr_interval_ms}
        # Shared manual ROI selection (x, y, w, h) or None
        self.roi_holder = {"roi": None}
        # Shared auto-scan toggle (False with --no-auto)
        self.auto_holder = {"enabled": not no_auto}

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
        logger.info(
            "Decoder availability: zbar=%s, OCR=%s",
            self._zbar_available, self._ocr_available,
        )

        # Results
        self.results_manager = ResultsManager()

        # --- Build UI ---
        self.preview_panel = PreviewPanel(
            self,
            on_selection_end=self._request_ocr_now,
        )
        self.preview_panel.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.results_panel = ResultsPanel(self)
        self.results_panel.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Status bar: text (left) + interactive toggle buttons (right)
        self.status_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.status_bar.grid(
            row=1, column=0, columnspan=2,
            padx=5, pady=(0, 2),
            sticky="ew",
        )
        self.status_bar.columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Initialising...",
            font=FONTS["small"],
            text_color="#888888",
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

        self.pause_btn = ctk.CTkButton(
            self.status_bar,
            text="Pause",
            font=FONTS["small"],
            width=90,
            height=24,
            command=self._toggle_pause,
        )
        self.pause_btn.grid(row=0, column=1, padx=(8, 4))

        self.auto_btn = ctk.CTkButton(
            self.status_bar,
            text="AUTO" if not no_auto else "MANUAL",
            font=FONTS["small"],
            width=90,
            height=24,
            fg_color=COLOURS["info"] if not no_auto else COLOURS["warning"],
            hover_color=COLOURS["info"] if not no_auto else COLOURS["warning"],
            command=self._toggle_auto,
        )
        self.auto_btn.grid(row=0, column=2, padx=(0, 4))

        # --- Start background threads ---
        self.capture_thread = CaptureThread(
            camera_index=camera_index,
            frame_queue=self.frame_queue,
            ocr_queue=self.ocr_queue,
            pause_event=self.pause_event,
            stop_event=self.stop_event,
            frame_interval_ms=frame_interval_ms,
            ocr_interval_ms=ocr_interval_ms,
            ocr_interval_holder=self.ocr_interval_holder,
            auto_holder=self.auto_holder,
            debug=debug,
        )
        self.capture_thread.start()

        self.ocr_thread = OCRThread(
            ocr_queue=self.ocr_queue,
            result_queue=self.ocr_result_queue,
            stop_event=self.stop_event,
            pause_event=self.pause_event,
            roi_holder=self.roi_holder,
            ocr_interval_holder=self.ocr_interval_holder,
            auto_holder=self.auto_holder,
            debug=debug,
        )
        self.ocr_thread.start()

        # --- Periodic UI update ---
        self._ui_fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._last_decay_time = 0.0
        self._last_results_version = -1
        self._refresh_status_buttons()
        self.after(50, self._update_loop)

        # --- Clean shutdown ---
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Keyboard shortcuts ---
        self.bind("<space>", lambda _e: self._toggle_pause())
        self.bind("<Escape>", lambda _e: self._on_close())
        self.bind("<Key-r>", lambda _e: self._clear_results())
        self.bind("<Key-a>", lambda _e: self._toggle_auto())

    # ------------------------------------------------------------------
    # UI update loop (runs on the main thread via .after())
    # ------------------------------------------------------------------

    def _update_loop(self) -> None:
        """Periodic update: poll queues, redraw preview, refresh results."""
        # Keep the OCR thread's ROI in sync with the UI selection
        self.roi_holder["roi"] = self.preview_panel.selection

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

            # 4. Decay stale results (once per second is plenty)
            now = time.time()
            if now - self._last_decay_time >= 1.0:
                self._last_decay_time = now
                self.results_manager.decay()

            # 5. Redraw preview
            self.preview_panel.update(
                self._latest_frame,
                self._latest_zbar,
                self.results_manager.get_active(),
            )

            # 6. Refresh results list only when something changed
            version = self.results_manager.version
            if version != self._last_results_version:
                self._last_results_version = version
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
            parts.append("paused")
        if self.no_auto and not self.roi_holder.get("roi"):
            parts.append("no region selected")

        if not self._zbar_available:
            parts.append("zbar unavailable")
        if not self._ocr_available:
            parts.append("OCR unavailable")

        self.status_label.configure(
            text=" \u00b7 ".join(parts) if parts else "No camera feed",
        )

        self.after(50, self._update_loop)

    # ------------------------------------------------------------------
    # App lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Enter the CTk main loop."""
        logger.info("Entering mainloop")
        self.mainloop()
        logger.info("Mainloop exited")

    def _toggle_pause(self) -> None:
        """Pause or resume scanning."""
        self._paused = not self._paused
        if self._paused:
            self.pause_event.set()
            logger.info("Scanning paused")
        else:
            self.pause_event.clear()
            logger.info("Scanning resumed")
        self._refresh_status_buttons()

    def _toggle_auto(self) -> None:
        """Switch between automatic scanning and manual-only mode."""
        enabled = not self.auto_holder.get("enabled")
        self.auto_holder["enabled"] = enabled
        self.no_auto = not enabled
        logger.info(
            "%s", "Auto-scan enabled" if enabled
            else "Auto-scan disabled (manual-only mode)",
        )
        self._refresh_status_buttons()

    def _refresh_status_buttons(self) -> None:
        """Sync the status-bar toggle buttons with current state."""
        if self._paused:
            self.pause_btn.configure(
                text="Resume", fg_color=COLOURS["warning"],
                hover_color=COLOURS["warning"],
            )
        else:
            self.pause_btn.configure(
                text="Pause", fg_color=COLOURS["info"],
                hover_color=COLOURS["info"],
            )
        if self.auto_holder.get("enabled"):
            self.auto_btn.configure(
                text="AUTO", fg_color=COLOURS["info"],
                hover_color=COLOURS["info"],
            )
        else:
            self.auto_btn.configure(
                text="MANUAL", fg_color=COLOURS["warning"],
                hover_color=COLOURS["warning"],
            )

    def _clear_results(self) -> None:
        """Clear all results (R key)."""
        count = self.results_manager.count
        self.results_manager.clear()
        logger.info("Cleared %d result(s)", count)

    def _request_ocr_now(self) -> None:
        """Run an immediate OCR pass on the latest frame (mouse-up on ROI)."""
        # Sync the ROI first: the update loop only does this every 50 ms,
        # and the OCR thread may consume the pushed frame before that.
        self.roi_holder["roi"] = self.preview_panel.selection

        frame = self.preview_panel.latest_frame
        if frame is None:
            return
        try:
            self.ocr_queue.put_nowait(frame.copy())
            logger.info("Immediate OCR pass requested (manual selection)")
        except queue.Full:
            logger.debug("ocr_queue full; immediate OCR pass skipped")

    def _on_close(self) -> None:
        """Clean up threads and close the window."""
        logger.info("Shutting down: signalling background threads")
        self.stop_event.set()
        try:
            self.destroy()
        except Exception:
            pass
        logger.info("Shutdown complete")