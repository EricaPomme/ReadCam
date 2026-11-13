"""Camera preview panel with bounding-box overlays and ROI selection."""

import logging
from typing import Callable, Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from ..camera import ZBarResult
from ..config import (
    COLOUR_QR,
    COLOUR_BARCODE,
    COLOUR_TEXT,
    COLOUR_UNKNOWN,
    PREVIEW_MAX_WIDTH,
    PREVIEW_MAX_HEIGHT,
)
from ..results import Result
from ..util import truncate
from .theme import COLOURS

logger = logging.getLogger(__name__)

# BGR colour for the manual ROI selection box (yellow)
COLOUR_SELECTION = (60, 220, 255)


class PreviewPanel(ctk.CTkFrame):
    """Panel that displays the live camera feed with detection overlays.

    Supports manual region selection: drag with the left mouse button
    to draw a box; the box persists from mouse-up until the next
    mouse-down on the display. The selected region (in frame
    coordinates) is exposed via :attr:`selection` for targeted OCR.
    """

    def __init__(self, master, on_selection_end: Optional[Callable] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=COLOURS["card_bg"])

        self.on_selection_end = on_selection_end

        self.image_label = ctk.CTkLabel(
            self,
            text="No camera feed\n(drag to select an OCR region)",
            font=("Segoe UI", 16),
            cursor="crosshair",
        )
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        self._current_ctk_image = None
        self._frame_size = (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)

        # Cached last render inputs, so drags can re-render without
        # waiting for a new frame from the capture thread.
        self._last_frame: Optional[np.ndarray] = None
        self._last_zbar: list[ZBarResult] = []
        self._last_active: list[Result] = []

        # Display mapping: frame coords -> displayed pixels
        self._render_scale = 1.0
        self._render_offset = (0, 0)

        # Selection state (in frame coordinates)
        self._selection: Optional[tuple] = None
        self._drag_start: Optional[tuple] = None   # frame coords
        self._drag_current: Optional[tuple] = None

        self.image_label.bind("<ButtonPress-1>", self._on_press)
        self.image_label.bind("<B1-Motion>", self._on_drag)
        self.image_label.bind("<ButtonRelease-1>", self._on_release)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update(
        self,
        frame: Optional[np.ndarray],
        zbar_results: list[ZBarResult],
        active_results: list[Result],
    ) -> None:
        """Refresh the preview with the latest frame and overlays."""
        if frame is None:
            return

        self._last_frame = frame
        self._last_zbar = zbar_results
        self._last_active = active_results
        self._render(frame, zbar_results, active_results)

    def _render(
        self,
        frame: np.ndarray,
        zbar_results: list[ZBarResult],
        active_results: list[Result],
    ) -> None:
        """Render a frame with all overlays (called on frame + on drags)."""
        display = frame.copy()

        # Draw zbar (QR / barcode) bounding boxes
        for zr in zbar_results:
            colour = self._colour_for_type(zr.type)
            if zr.points:
                pts = np.array(zr.points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, colour, 2)
                label = f"{zr.type}: {truncate(zr.data, 30)}"
                cv2.putText(
                    display, label,
                    (zr.points[0][0], zr.points[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
                )
            elif zr.rect:
                x, y, w, h = zr.rect
                cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)
                label = f"{zr.type}: {truncate(zr.data, 30)}"
                cv2.putText(
                    display, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
                )

        # Draw OCR results that aren't already covered by zbar
        zbar_keys = {(zr.type.upper(), zr.data) for zr in zbar_results}
        for result in active_results:
            if result.bbox and (result.type, result.value) not in zbar_keys:
                x, y, w, h = result.bbox
                colour = self._colour_for_type(result.type)
                cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)
                label = f"{result.type}: {truncate(result.value, 30)}"
                cv2.putText(
                    display, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
                )

        # Draw the persistent selection and/or the active drag box
        if self._dragging:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_current
            box = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        else:
            box = self._selection
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(display, (x, y), (x + w, y + h), COLOUR_SELECTION, 2)
            cv2.putText(
                display, "OCR region", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_SELECTION, 2,
            )

        # Convert BGR OpenCV -> RGB PIL -> CTkImage
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image = self._resize_to_fit(pil_image)

        ctk_image = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=pil_image.size,
        )

        self.image_label.configure(image=ctk_image, text="")
        self._current_ctk_image = ctk_image

        # Remember the mapping for mouse-coordinate conversion.
        # CTk displays the photo on the *inner* tkinter.Label (scaled by
        # the widget DPI factor), and that inner label receives mouse
        # events -- so the drawn image size comes from the inner label,
        # not from our PIL image or the outer widget.
        inner = getattr(self.image_label, "_label", None)
        if inner is not None:
            inner_w = inner.winfo_width()
            inner_h = inner.winfo_height()
            if inner_w > 10 and inner_h > 10:
                self._render_scale = inner_w / frame.shape[1]
                self._render_offset = (0, 0)

    def _resize_to_fit(self, image: Image.Image) -> Image.Image:
        """Scale the image to fit the preview area while preserving aspect ratio."""
        max_w, max_h = self._frame_size

        label_w = self.image_label.winfo_width() or max_w
        label_h = self.image_label.winfo_height() or max_h

        if label_w < 10 or label_h < 10:
            label_w, label_h = max_w, max_h

        img_w, img_h = image.size
        scale = min(label_w / img_w, label_h / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        if new_w != img_w or new_h != img_h:
            image = image.resize((new_w, new_h), Image.BILINEAR)

        return image

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    @property
    def _dragging(self) -> bool:
        return self._drag_start is not None and self._drag_current is not None

    @property
    def selection(self) -> Optional[tuple]:
        """The persisted selection as (x, y, w, h) in frame coordinates."""
        return self._selection

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """The most recently displayed frame (for immediate OCR passes)."""
        return self._last_frame

    def _to_frame_coords(self, event) -> Optional[tuple]:
        """Convert a mouse event to frame coordinates.

        Events may arrive from the inner tkinter.Label (which displays
        the image and fills it exactly) or from the surrounding canvas
        (clicks that miss the image). Coordinates are normalised to be
        relative to the drawn image before applying the scale.
        """
        frame = self._last_frame
        if frame is None:
            return None

        inner = getattr(self.image_label, "_label", None)
        canvas = getattr(self.image_label, "_canvas", None)

        if inner is not None and event.widget is inner:
            # Already relative to the drawn image
            ex, ey = event.x, event.y
        elif canvas is not None and event.widget is canvas and inner is not None:
            # Shift by the inner label's position within the canvas
            ex = event.x - inner.winfo_x()
            ey = event.y - inner.winfo_y()
        else:
            ex, ey = event.x, event.y

        x = int(ex / self._render_scale)
        y = int(ey / self._render_scale)

        fh, fw = frame.shape[:2]
        return (max(0, min(x, fw - 1)), max(0, min(y, fh - 1)))

    def _on_press(self, event) -> None:
        """Mouse-down: clear any persisted selection and start a drag."""
        point = self._to_frame_coords(event)
        if point is None:
            return
        if self._selection is not None:
            logger.info("Cleared manual OCR selection")
        self._selection = None
        self._drag_start = point
        self._drag_current = point

    def _on_drag(self, event) -> None:
        point = self._to_frame_coords(event)
        if point is None or self._drag_start is None:
            return
        self._drag_current = point
        # Live re-render using the cached frame
        self._render(self._last_frame, self._last_zbar, self._last_active)

    def _on_release(self, event) -> None:
        """Mouse-up: commit the drag as a persistent selection (if big enough)."""
        point = self._to_frame_coords(event)
        if point is None or self._drag_start is None:
            return

        x0, y0 = self._drag_start
        x1, y1 = point
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        self._drag_start = None
        self._drag_current = None

        # Too-small drags are treated as a click (clear the selection)
        if w < 10 or h < 10:
            if self._selection is not None:
                logger.info("Cleared manual OCR selection (click)")
            self._selection = None
            self._render(self._last_frame, self._last_zbar, self._last_active)
            return

        self._selection = (x, y, w, h)
        logger.info(
            "Manual OCR selection set: (%d, %d, %d, %d)", x, y, w, h,
        )
        self._render(self._last_frame, self._last_zbar, self._last_active)

        if self.on_selection_end:
            self.on_selection_end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _colour_for_type(det_type: str) -> tuple[int, int, int]:
        """Map a detection type to its BGR colour."""
        if det_type == "QR":
            return COLOUR_QR
        elif det_type in (
            "EAN-13", "EAN-8", "UPC-A", "UPC-E",
            "CODE128", "CODE39", "CODE93", "I2OF5",
            "PDF417", "CODABAR", "DATABAR",
        ):
            return COLOUR_BARCODE
        elif det_type == "TXT":
            return COLOUR_TEXT
        else:
            return COLOUR_UNKNOWN