"""Main ReadCam application window built with CustomTkinter."""

import logging
import queue
import threading
import time
from typing import Optional

import customtkinter as ctk
import numpy as np

logger = logging.getLogger(__name__)


def _log_thread_exception(args) -> None:
    """Log uncaught thread exceptions instead of letting them vanish."""
    logger.critical(
        "Uncaught exception in thread %s",
        args.thread.name if args.thread else "<unknown>",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


threading.excepthook = _log_thread_exception

from ..capture import copy_image_to_clipboard, save_frame
from ..camera import CaptureThread, CameraDiscovery, OCRThread
from ..config import (
    DEFAULT_FLIP,
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_OCR_INTERVAL,
    DEFAULT_ROTATION,
    UI_MIN_HEIGHT,
    UI_MIN_WIDTH,
    UI_PREVIEW_MIN_WIDTH,
    UI_RESULTS_MIN_WIDTH,
    UI_TITLE,
)
from ..decoders.ocr_decoder import OCRDecoder
from ..decoders.zbar_decoder import ZBarDecoder
from ..results import ResultsManager
from ..util import log_event
from .capture_panel import CapturePanel
from .filter_panel import FilterPanel
from .preview import PreviewPanel, draw_overlays
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
        no_ocr: bool = False,
        rotation: int = DEFAULT_ROTATION,
        flip: str = DEFAULT_FLIP,
        debug: bool = False,
    ):
        setup_theme()
        super().__init__()

        self.title(UI_TITLE)
        self.minsize(UI_MIN_WIDTH, UI_MIN_HEIGHT)

        # Grid: column 0 = toolbar / preview / capture / filters, column
        # 1 = drag sash, column 2 = results. Column 0 takes the extra
        # space; column 2 stays at its natural (minimum) width so the
        # results pane only grows when dragged wider via the sash.
        self.grid_columnconfigure(0, weight=1, minsize=UI_PREVIEW_MIN_WIDTH)
        self.grid_columnconfigure(1, weight=0, minsize=4)
        self.grid_columnconfigure(2, weight=0, minsize=UI_RESULTS_MIN_WIDTH)
        self.grid_rowconfigure(0, weight=0)  # camera toolbar
        self.grid_rowconfigure(1, weight=1)  # preview
        self.grid_rowconfigure(2, weight=0)  # capture panel
        self.grid_rowconfigure(3, weight=0)  # filter panel
        self.grid_rowconfigure(4, weight=0)  # status bar

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
        # Shared OCR bypass toggle (False with --no-ocr; barcodes only)
        self.ocr_holder = {"enabled": not no_ocr}
        # Shared live filter thresholds written by the sliders
        self.filter_holder: dict = {}
        # Shared camera index; CaptureThread re-opens on change
        self.camera_index_holder: dict = {"index": camera_index}
        # Shared display transform (rotation degrees / flip mode)
        self.transform_holder: dict = {"rotation": rotation, "flip": flip}

        # Thread control
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self._paused = False
        self._sash_dragging = False

        # State
        self._latest_frame: Optional = None
        self._latest_zbar: list = []
        # Transient status-bar message (capture feedback) with expiry
        self._status_message: str = ""
        self._status_message_until: float = 0.0

        # Decoder availability (checked once at startup)
        self._ocr_available = OCRDecoder._check_available()
        self._zbar_available = ZBarDecoder._check_available()
        logger.info(
            "Decoder availability: zbar=%s, OCR=%s",
            self._zbar_available, self._ocr_available,
        )

        # Results (shares the filter holder with the OCR thread)
        self.results_manager = ResultsManager(filter_holder=self.filter_holder)

        # --- Build UI ---
        # Toolbar row (above the preview): camera selector + refresh
        self.toolbar = ctk.CTkFrame(self, fg_color="transparent")
        self.toolbar.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="ew")
        self.toolbar.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.toolbar, text="Camera:", font=FONTS["small"],
        ).grid(row=0, column=0, padx=(0, 4))

        self._camera_map: dict[str, int] = {}
        self._cameras: list = []
        self._failure_handled = False
        self.camera_menu = ctk.CTkOptionMenu(
            self.toolbar,
            values=["Detecting cameras..."],
            font=FONTS["small"],
            height=24,
            state="disabled",
            command=self._on_camera_selected,
        )
        self.camera_menu.grid(row=0, column=1, sticky="ew")

        self.camera_refresh_btn = ctk.CTkButton(
            self.toolbar,
            text="Refresh",
            font=FONTS["small"],
            width=70,
            height=24,
            command=self._refresh_cameras,
        )
        self.camera_refresh_btn.grid(row=0, column=2, padx=(4, 0))

        self.rotate_btn = ctk.CTkButton(
            self.toolbar,
            text=f"Rotate: {self.transform_holder['rotation']}\u00b0",
            font=FONTS["small"],
            width=90,
            height=24,
            command=self._on_rotate,
        )
        self.rotate_btn.grid(row=0, column=3, padx=(4, 0))

        _flip_labels = {"none": "Off", "h": "H", "v": "V", "hv": "HV"}
        self.flip_btn = ctk.CTkButton(
            self.toolbar,
            text=f"Flip: {_flip_labels.get(self.transform_holder['flip'], 'Off')}",
            font=FONTS["small"],
            width=90,
            height=24,
            command=self._on_flip,
        )
        self.flip_btn.grid(row=0, column=4, padx=(4, 0))

        self.preview_panel = PreviewPanel(
            self,
            on_selection_end=self._on_manual_selection_end,
        )
        self.preview_panel.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.capture_panel = CapturePanel(
            self,
            on_disk=self._on_capture_disk,
            on_clip=self._on_capture_clip,
        )
        self.capture_panel.grid(row=2, column=0, padx=5, pady=(0, 5), sticky="ew")

        self.filter_panel = FilterPanel(
            self,
            filters=self.filter_holder,
            ocr_holder=self.ocr_holder,
            on_change_end=self._on_filters_changed,
        )
        self.filter_panel.grid(row=3, column=0, padx=5, pady=(0, 5), sticky="ew")

        self.results_panel = ResultsPanel(self)
        self.results_panel.grid(
            row=0, column=2, rowspan=4, padx=5, pady=5, sticky="nsew",
        )

        # Drag sash between preview and results columns: dragging it
        # changes column 0's minsize, which the gridded panels follow
        # via sticky="ew"/"nsew".
        self.sash = ctk.CTkFrame(
            self, width=4, corner_radius=0, fg_color=COLOURS["border"],
        )
        self.sash.grid(row=0, column=1, rowspan=4, sticky="ns", pady=5)
        self.sash.bind("<ButtonPress-1>", self._sash_drag_start)
        self.sash.bind("<B1-Motion>", self._sash_drag)
        self.sash.bind("<ButtonRelease-1>", self._sash_drag_end)
        self.sash.bind("<Enter>", lambda _e: self._set_sash_cursor("sb_h_double_arrow"))
        self.sash.bind("<Leave>", lambda _e: self._set_sash_cursor(""))

        # Status bar: text (left) + interactive toggle buttons (right)
        self.status_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.status_bar.grid(
            row=4, column=0, columnspan=3,
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
            camera_index_holder=self.camera_index_holder,
            frame_queue=self.frame_queue,
            ocr_queue=self.ocr_queue,
            pause_event=self.pause_event,
            stop_event=self.stop_event,
            frame_interval_ms=frame_interval_ms,
            ocr_interval_ms=ocr_interval_ms,
            ocr_interval_holder=self.ocr_interval_holder,
            auto_holder=self.auto_holder,
            ocr_holder=self.ocr_holder,
            transform_holder=self.transform_holder,
            filter_holder=self.filter_holder,
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
            ocr_holder=self.ocr_holder,
            filter_holder=self.filter_holder,
            debug=debug,
        )
        self.ocr_thread.start()

        # --- Periodic UI update ---
        self._ui_fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._last_decay_time = 0.0
        self._last_results_refresh = 0.0
        self._last_results_version = -1
        self._refresh_status_buttons()
        self.after(50, self._update_loop)

        # Populate the camera dropdown without blocking the window
        self._start_discovery(select_index=camera_index)

        # --- Clean shutdown ---
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Keyboard shortcuts ---
        self.bind("<space>", lambda _e: self._toggle_pause())
        self.bind("<Escape>", lambda _e: self._on_close())
        self.bind("<Key-r>", lambda _e: self._clear_results())
        self.bind("<Key-a>", lambda _e: self._toggle_auto())

    # ------------------------------------------------------------------
    # Layout sash
    # ------------------------------------------------------------------

    def _set_sash_cursor(self, cursor: str) -> None:
        """Show a resize cursor while hovering the sash."""
        self.sash.configure(cursor=cursor)

    def _sash_drag_start(self, _event) -> None:
        """Pause detection while the sash is dragged.

        The capture thread skips decoding (and starves OCR) whenever
        ``pause_event`` is set, freeing CPU for the resize re-renders.
        The preview keeps rendering its cached frame, so the resize
        itself stays live. An explicit user Pause is respected: the
        drag never resumes scanning on release.
        """
        self._sash_dragging = not self._paused
        if self._sash_dragging:
            self.pause_event.set()

    def _sash_drag_end(self, _event) -> None:
        """Resume detection after a sash drag (unless user-paused)."""
        if self._sash_dragging:
            self._sash_dragging = False
            if not self._paused:
                self.pause_event.clear()

    def _sash_drag(self, _event) -> None:
        """Resize column 0 (preview side) while the sash is dragged."""
        # Pointer position relative to the window, minus a small margin
        # so the sash itself stays visible.
        x = self.winfo_pointerx() - self.winfo_rootx()
        max_x = self.winfo_width() - UI_RESULTS_MIN_WIDTH - 20
        width = max(UI_PREVIEW_MIN_WIDTH, min(int(x), max_x))
        self.grid_columnconfigure(0, minsize=width)

    # ------------------------------------------------------------------
    # UI update loop (runs on the main thread via .after())
    # ------------------------------------------------------------------

    def _update_loop(self) -> None:
        """Periodic update: poll queues, redraw preview, refresh results."""
        try:
            self._update_tick()
        except Exception:
            # Never let a single bad tick kill the loop (a dead loop
            # stops consuming frame_queue and the UI freezes).
            logger.exception("UI update tick failed")
        finally:
            self.after(50, self._update_loop)

    def _update_tick(self) -> None:
        """One iteration of the update loop."""
        # Keep the OCR thread's ROI in sync with the UI selection
        self.roi_holder["roi"] = self.preview_panel.selection

        if not self._paused:
            # 1. Consume latest frame from capture thread
            try:
                while True:
                    self._latest_frame, self._latest_zbar = (
                        self.frame_queue.get_nowait()
                    )
            except queue.Empty:
                pass

            # 2. Consume OCR results
            try:
                while True:
                    text_detections = self.ocr_result_queue.get_nowait()
                    for det in text_detections:
                        if det.type == "TXT":
                            self.results_manager.add_text_hit(
                                det.value, det.bbox, det.confidence,
                            )
                        else:
                            # Barcode/QR results decoded in the OCR thread
                            # (zxing/zbar) keep their symbology type.
                            self.results_manager.add_or_update(
                                det.type, det.value, det.bbox,
                            )
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

            # 6. Refresh results list when something changed, or once per
            # second so confidence-driven re-sorting and relative times
            # stay current (version only bumps on add/remove).
            now_refresh = time.time()
            version = self.results_manager.version
            if (
                version != self._last_results_version
                or now_refresh - self._last_results_refresh >= 1.0
            ):
                self._last_results_version = version
                self._last_results_refresh = now_refresh
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
        self._sync_camera_names()
        self._sync_camera_failure()
        parts: list[str] = []

        # Transient capture feedback takes precedence while active
        if self._status_message and time.time() < self._status_message_until:
            parts.append(self._status_message)
        else:
            if self._status_message:
                self._status_message = ""

        # Camera status
        t = self.capture_thread
        if t.connected:
            parts.append(t.current_index_name)
        elif t.open_failed:
            parts.append(f"Camera {t.camera_index} unavailable")
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
        if not self.ocr_holder.get("enabled"):
            parts.append("OCR off")
        elif not self._ocr_available:
            parts.append("OCR unavailable")

        self.status_label.configure(
            text=" \u00b7 ".join(parts) if parts else "No camera feed",
        )

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

    # ------------------------------------------------------------------
    # View transform (rotation / flip)
    # ------------------------------------------------------------------

    def _on_rotate(self) -> None:
        """Cycle the view rotation: 0 -> 90 -> 180 -> 270 degrees."""
        rotation = (self.transform_holder.get("rotation", 0) + 90) % 360
        self.transform_holder["rotation"] = rotation
        self.rotate_btn.configure(text=f"Rotate: {rotation}\u00b0")
        logger.info("View rotation set to %d degrees", rotation)
        self._on_transform_changed()

    def _on_flip(self) -> None:
        """Cycle the view flip mode: Off -> H -> V -> HV."""
        modes = ["none", "h", "v", "hv"]
        labels = {"none": "Off", "h": "H", "v": "V", "hv": "HV"}
        current = self.transform_holder.get("flip", "none")
        flip = modes[(modes.index(current) + 1) % len(modes)]
        self.transform_holder["flip"] = flip
        self.flip_btn.configure(text=f"Flip: {labels[flip]}")
        logger.info("View flip set to %s", flip)
        self._on_transform_changed()

    def _on_transform_changed(self) -> None:
        """View transform changed: drop stale results and selection.

        Old bounding boxes and the manual ROI no longer match the new
        frame geometry (same reasoning as a camera switch).
        """
        self._clear_results()
        self.preview_panel.clear()
        self._latest_frame = None
        self._latest_zbar = []

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

    def _on_filters_changed(self) -> None:
        """Filter sliders changed: drop invalidated results, then re-OCR."""
        removed = self.results_manager.revalidate_text()
        if removed:
            self.results_panel.update_results(
                self.results_manager.get_sorted(),
            )
        self._request_ocr_now()

    def _on_manual_selection_end(self) -> None:
        """Manual ROI selection committed: clear stale results, then OCR."""
        self._clear_results()
        self._request_ocr_now()

    # ------------------------------------------------------------------
    # Capture saving
    # ------------------------------------------------------------------

    def _flash_status(self, message: str) -> None:
        """Show a transient message in the status bar (5 s)."""
        self._status_message = message
        self._status_message_until = time.time() + 5.0
        logger.info("Status: %s", message)

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Compose the frame to capture: raw, or with overlays if BB is on."""
        frame, zbar, active = self.preview_panel.render_inputs()
        if frame is None:
            return None
        if not self.capture_panel.include_boxes.get():
            return frame.copy()
        display = frame.copy()
        draw_overlays(display, zbar, active, self.preview_panel.selection)
        return display

    def _on_capture_disk(self) -> None:
        """Save the current frame as a PNG in the output directory."""
        frame = self._capture_frame()
        if frame is None:
            self._flash_status("No camera feed to capture")
            return
        try:
            path = save_frame(frame, self.capture_panel.output_dir)
            log_event("Capture saved to %s", path)
            self._flash_status(f"Saved to {path}")
        except Exception as exc:
            logger.exception("Capture save failed")
            self._flash_status(f"Save failed: {exc}")

    def _on_capture_clip(self) -> None:
        """Copy the current frame image to the clipboard."""
        frame = self._capture_frame()
        if frame is None:
            self._flash_status("No camera feed to capture")
            return
        if copy_image_to_clipboard(frame):
            log_event("Capture copied to clipboard")
            self._flash_status("Copied to clipboard")
        else:
            self._flash_status("Clipboard image copy failed (not supported off Windows)")

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

    # ------------------------------------------------------------------
    # Camera selection
    # ------------------------------------------------------------------

    def _start_discovery(
        self,
        select_index: Optional[int] = None,
        full: bool = False,
        apply_selection: bool = False,
    ) -> None:
        """Populate the camera menu.

        By default this is instant and lazy: indices 0-9 with generic
        names, no device access. With *full=True* (Refresh button or a
        failed selection) a real probe runs in a background thread and
        returns only the cameras that actually exist, with friendly
        names. With *apply_selection=True* the chosen camera is also
        written to the holder (used after a failure, to resume
        scanning on the first available camera).
        """
        if not full:
            self._populate_camera_menu(
                CameraDiscovery.quick_indices(), select_index, apply_selection,
            )
            return

        self.camera_menu.configure(
            values=["Detecting cameras..."], state="disabled",
        )
        self._camera_map = {}

        def probe() -> None:
            cameras = CameraDiscovery.list_cameras()
            self.after(0, lambda: self._populate_camera_menu(
                cameras, select_index, apply_selection,
            ))

        threading.Thread(target=probe, daemon=True, name="camprobe").start()

    def _refresh_cameras(self) -> None:
        """Re-run camera discovery (Refresh button; full device probe)."""
        logger.info("Refreshing camera list")
        self._start_discovery(
            select_index=self.camera_index_holder.get("index"),
            full=True,
        )

    def _populate_camera_menu(
        self,
        cameras: list,
        select_index: Optional[int] = None,
        apply_selection: bool = False,
    ) -> None:
        """Fill the dropdown with detected cameras (runs on the UI thread)."""
        if not cameras:
            self.camera_menu.configure(
                values=["No cameras found"], state="disabled",
            )
            logger.warning("No cameras detected")
            return

        self._cameras = list(cameras)
        self._camera_map = {
            f"[{cam.index}] {cam.name}": cam.index for cam in cameras
        }
        choices = list(self._camera_map)
        self.camera_menu.configure(values=choices, state="normal")

        # Prefer the requested index (current or --camera), else the first
        chosen = next(
            (label for label, idx in self._camera_map.items()
             if idx == select_index),
            choices[0],
        )
        self.camera_menu.set(chosen)
        logger.info("Camera menu populated: %s", choices)

        if apply_selection:
            self._on_camera_selected(chosen)

    def _on_camera_selected(self, choice: str) -> None:
        """Dropdown selection: request a live switch via the shared holder."""
        index = self._camera_map.get(choice)
        if index is None or index == self.camera_index_holder.get("index"):
            return
        self.camera_index_holder["index"] = index
        logger.info("Camera selection changed to [%d] %s", index, choice)
        # Stale frames/selection/cache from the previous camera no longer
        # apply; the preview stays blank until the new camera delivers.
        self.preview_panel.clear()
        self._latest_frame = None
        self._latest_zbar = []

    def _sync_camera_names(self) -> None:
        """Fold in friendly names resolved by the capture thread.

        The lazy startup list only knows generic names; once a camera
        is opened, its real name (probed in the capture thread) replaces
        the generic entry in the dropdown.
        """
        t = self.capture_thread
        if not t.connected or t._camera_name is None:
            return
        for cam in self._cameras:
            if cam.index == t.camera_index and cam.name != t._camera_name:
                cam.name = t._camera_name
                self._camera_map = {
                    f"[{c.index}] {c.name}": c.index for c in self._cameras
                }
                self.camera_menu.configure(values=list(self._camera_map))
                self.camera_menu.set(f"[{cam.index}] {cam.name}")
                logger.info("Camera [%d] resolved as %r", cam.index, cam.name)
                break

    def _sync_camera_failure(self) -> None:
        """React to a selection that failed to open (runs on the UI thread).

        When the capture thread gives up on an index (3 failed open
        attempts): drop that entry from the dropdown, then run one
        background full probe so the list converges to the cameras
        that actually exist -- no Refresh needed. The probe result
        auto-selects the first available camera to resume scanning.
        """
        t = self.capture_thread
        if t.connected:
            self._failure_handled = False
            return
        if t.gave_up and not self._failure_handled:
            self._failure_handled = True
            index = t.camera_index
            logger.info(
                "Camera [%d] unavailable; removing from list and re-probing",
                index,
            )
            self._remove_camera(index)
            self._start_discovery(full=True, apply_selection=True)

    def _remove_camera(self, index: int) -> None:
        """Drop an index from the dropdown (proved unavailable)."""
        self._cameras = [c for c in self._cameras if c.index != index]
        if not self._cameras:
            self.camera_menu.configure(
                values=["No cameras found"], state="disabled",
            )
            return
        self._camera_map = {
            f"[{c.index}] {c.name}": c.index for c in self._cameras
        }
        self.camera_menu.configure(
            values=list(self._camera_map), state="normal",
        )
        # Keep the failed selection visible (annotated) as the display
        # value; the probe result will replace it shortly.
        self.camera_menu.set(f"[{index}] Camera {index} (unavailable)")