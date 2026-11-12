"""Camera preview panel with bounding-box overlays."""

from typing import Optional

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


class PreviewPanel(ctk.CTkFrame):
    """Panel that displays the live camera feed with detection overlays."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=COLOURS["card_bg"])

        self.image_label = ctk.CTkLabel(
            self,
            text="No camera feed",
            font=("Segoe UI", 16),
        )
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        self._current_ctk_image = None
        self._frame_size = (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)

    def update(
        self,
        frame: Optional[np.ndarray],
        zbar_results: list[ZBarResult],
        active_results: list[Result],
    ) -> None:
        """Refresh the preview with the latest frame and overlays."""
        if frame is None:
            return

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
            image = image.resize((new_w, new_h), Image.LANCZOS)

        return image

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