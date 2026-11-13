"""Camera preview panel with bounding-box overlays and ROI selection."""

import logging
from typing import Callable, Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

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


def draw_overlays(
    display: np.ndarray,
    zbar_results: list[ZBarResult],
    active_results: list[Result],
    selection_box: Optional[tuple] = None,
) -> None:
    """Draw detection boxes, labels, and the OCR selection box on *display*.

    Shared by the live preview and capture saving, so both render the
    identical overlay pipeline. *display* is modified in place.
    """
    # Zbar (QR / barcode) bounding boxes
    for zr in zbar_results:
        colour = _colour_for_type(zr.type)
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

    # OCR results that aren't already covered by zbar
    zbar_keys = {(zr.type.upper(), zr.data) for zr in zbar_results}
    for result in active_results:
        if result.bbox and (result.type, result.value) not in zbar_keys:
            x, y, w, h = result.bbox
            colour = _colour_for_type(result.type)
            cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)
            label = f"{result.type}: {truncate(result.value, 30)}"
            cv2.putText(
                display, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
            )

    # Persistent selection and/or active drag box
    if selection_box is not None:
        x, y, w, h = selection_box
        cv2.rectangle(display, (x, y), (x + w, y + h), COLOUR_SELECTION, 2)
        cv2.putText(
            display, "OCR region", (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_SELECTION, 2,
        )


def _colour_for_type(det_type: str) -> tuple[int, int, int]:
    """Map a detection type to its BGR colour."""
    if det_type == "QR":
        return COLOUR_QR
    elif det_type in (
        "DM", "AZTEC", "SQCODE",
        "EAN-13", "EAN-8", "UPC-A", "UPC-E",
        "CODE128", "CODE39", "CODE93", "I2OF5",
        "PDF417", "CODABAR", "DATABAR",
    ):
        return COLOUR_BARCODE
    elif det_type == "TXT":
        return COLOUR_TEXT
    else:
        return COLOUR_UNKNOWN


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

        # Freeze the panel's requested size at its configured dimensions.
        # The displayed image fills the pane (up *or* down), so letting
        # the image drive the panel's request would pin the grid column
        # at the image size and stop the layout sash from shrinking the
        # preview. With propagation off, the sash's column minsize fully
        # controls the width; the label fills whatever is allocated and
        # the image recomputes to match on the next frame.
        self.pack_propagate(False)

        self.on_selection_end = on_selection_end

        self.image_label = ctk.CTkLabel(
            self,
            text="No camera feed\n(drag to select an OCR region)",
            font=("Segoe UI", 16),
            cursor="crosshair",
        )
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        self._current_photo_image: Optional[ImageTk.PhotoImage] = None
        self._showing_image = False
        self._frame_size = (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)

        # Cached last render inputs, so drags can re-render without
        # waiting for a new frame from the capture thread.
        self._last_frame: Optional[np.ndarray] = None
        self._last_zbar: list[ZBarResult] = []
        self._last_active: list[Result] = []

        # Display mapping: event pixels -> frame pixels, plus the
        # letterbox offset in frame pixels
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
        draw_overlays(display, zbar_results, active_results, self._selection_box())

        # Convert BGR OpenCV -> RGB PIL, scale to fit, and display on a
        # single reusable PhotoImage. Creating a new image per frame
        # churns thousands of Tk images; when one is garbage-collected
        # while the label still points at it, Tk raises TclError
        # ("image ... doesn't exist"). Updating one stable image via
        # paste() avoids that entirely.
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image, canvas_offset, fit_scale = self._compose_display(pil_image)

        if (
            self._current_photo_image is None
            or self._current_photo_image.width() != pil_image.width
            or self._current_photo_image.height() != pil_image.height
        ):
            # Size changed (or first frame): new PhotoImage
            self._current_photo_image = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(
                image=self._current_photo_image, text="",
            )
        else:
            self._current_photo_image.paste(pil_image)
            if not self._showing_image:
                self.image_label.configure(
                    image=self._current_photo_image, text="",
                )
        self._showing_image = True

        # Remember the mapping for mouse-coordinate conversion, built
        # from two independently measurable stages so DPI scaling can
        # never skew one without the other:
        #
        #   frame --fit_scale--> canvas --display_scale--> events
        #
        # The canvas is the letterboxed image we just built (pure pixel
        # arithmetic). The display scale is whatever factor CTk applies
        # between the canvas and the inner label's event space, measured
        # empirically as inner_width / canvas_width -- the same trick
        # that made the pre-letterbox mapping DPI-safe. Storing the
        # combined scale plus the letterbox offset converted into frame
        # pixels keeps _to_frame_coords a simple two-term inverse.
        inner = getattr(self.image_label, "_label", None)
        if inner is not None:
            inner_w = inner.winfo_width()
            inner_h = inner.winfo_height()
            if inner_w > 10 and inner_h > 10:
                display_scale = inner_w / pil_image.width
                self._render_scale = display_scale * fit_scale
                # Offsets in whole frame pixels -- cv2 drawing and the
                # ROI pipeline expect ints, and sub-pixel precision is
                # meaningless at frame resolution.
                self._render_offset = (
                    int(round(canvas_offset[0] / fit_scale)),
                    int(round(canvas_offset[1] / fit_scale)),
                )

    def _compose_display(
        self, image: Image.Image,
    ) -> tuple[Image.Image, tuple[int, int], float]:
        """Scale the frame to fill the preview label, letterboxed on black.

        The frame is scaled up or down (aspect preserved) to fit the
        label's allocated size and pasted centred onto a black canvas of
        exactly that size. Returning a canvas that always fills the
        label restores the "image fills the label" invariant the mouse
        mapping relies on, and the frame's position within the canvas
        is pure image-pixel arithmetic -- no DPI-scaled widget geometry.

        Returns the display image, the frame's ``(x, y)`` offset inside
        it (display-image pixels), and the frame-to-canvas fit scale.
        """
        max_w, max_h = self._frame_size

        label_w = self.image_label.winfo_width() or max_w
        label_h = self.image_label.winfo_height() or max_h

        if label_w < 10 or label_h < 10:
            label_w, label_h = max_w, max_h

        img_w, img_h = image.size
        scale = min(label_w / img_w, label_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        fit_scale = new_w / img_w

        if new_w != img_w or new_h != img_h:
            image = image.resize((new_w, new_h), Image.BILINEAR)

        offset = ((label_w - new_w) // 2, (label_h - new_h) // 2)
        if offset != (0, 0):
            canvas = Image.new("RGB", (label_w, label_h), (0, 0, 0))
            canvas.paste(image, offset)
            return canvas, offset, fit_scale
        return image, (0, 0), fit_scale

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

    def clear(self) -> None:
        """Reset cached frame, overlays, and any selection (camera switch)."""
        self._last_frame = None
        self._last_zbar = []
        self._last_active = []
        self._selection = None
        self._drag_start = None
        self._drag_current = None
        self._showing_image = False
        # Detach the image from the inner label *while the PhotoImage is
        # still referenced here*; the stable image object is kept for
        # reuse by the next frame.
        inner = getattr(self.image_label, "_label", None)
        if inner is not None:
            inner.configure(image="")
        self.image_label.configure(
            image=None,
            text="No camera feed\n(drag to select an OCR region)",
        )

    def _to_frame_coords(self, event) -> Optional[tuple]:
        """Convert a mouse event to frame coordinates.

        Events may arrive from the inner tkinter.Label (which displays
        the image and fills it exactly) or from the surrounding canvas
        (clicks that miss the image). Coordinates are normalised to be
        relative to the drawn image, then the combined display/fit
        scale and the letterbox offset (both recorded by ``_render``)
        map them into frame coordinates.
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

        # Event space -> display-image pixels -> frame pixels, via the
        # combined display/fit scale and the letterbox offset (in frame
        # pixels) recorded by _render.
        off_x, off_y = self._render_offset
        x = int(ex / self._render_scale) - off_x
        y = int(ey / self._render_scale) - off_y

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

    def _selection_box(self) -> Optional[tuple]:
        """The active drag box or the persisted selection, if any."""
        if self._dragging:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_current
            return (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        return self._selection

    def render_inputs(self) -> tuple[Optional[np.ndarray], list, list]:
        """The cached frame, zbar results, and active results.

        Capture saving uses these to render the identical overlay
        pipeline as the preview, without keeping a second copy of the
        state in the app.
        """
        return self._last_frame, self._last_zbar, self._last_active