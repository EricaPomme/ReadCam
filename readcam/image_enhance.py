"""Live image adjustments driven by the shared filter holder.

The :class:`ImageEnhancer` applies the user's zoom (digital centre
crop), colour-mode and adjustment sliders (contrast, saturation,
median blur, sharpness) to a BGR frame in the capture thread. Each
stage is skipped when set to its neutral value so the per-frame cost
stays near zero when the user has not touched anything.
"""

import logging

import cv2
import numpy as np

from .config import (
    COLOUR_MODE_1BIT,
    COLOUR_MODE_COLOUR,
    COLOUR_MODE_GREYSCALE,
    IMAGE_CONTRAST,
    IMAGE_MEDIAN_BLUR,
    IMAGE_SATURATION,
    IMAGE_SHARPNESS,
    IMAGE_UNSHARP_SIGMA,
    IMAGE_ZOOM,
)

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Apply filter-holder image adjustments to a BGR frame.

    *filters* is the shared ``filter_holder`` dict written by the
    Filters panel (zoom, colour mode, contrast, saturation, median
    blur, sharpness). Missing keys fall back to the config defaults,
    so the enhancer is safe to use before the panel has been built.
    """

    def __init__(self, filters: dict):
        self._filters = filters

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return the adjusted frame; *frame* is not modified."""
        filters = self._filters

        # 0. Digital zoom: centre-crop to 1/zoom then resize back to
        # the original size so every consumer (preview, zbar, OCR,
        # captures) sees identical geometry and coordinates. Skipped
        # at 1.0x so the default per-frame cost stays zero.
        zoom = float(filters.get("zoom", IMAGE_ZOOM))
        if zoom > 1.001:
            height, width = frame.shape[:2]
            crop_w = max(1, int(round(width / zoom)))
            crop_h = max(1, int(round(height / zoom)))
            x0 = (width - crop_w) // 2
            y0 = (height - crop_h) // 2
            frame = frame[y0:y0 + crop_h, x0:x0 + crop_w]
            frame = cv2.resize(
                frame, (width, height), interpolation=cv2.INTER_LINEAR,
            )

        # 1. Colour mode: display-only conversion. Greyscale and
        # 1-bit modes are folded back to 3 channels so overlays and
        # the preview pipeline stay unchanged.
        mode = filters.get("colour_mode", "Colour")
        if mode == COLOUR_MODE_GREYSCALE:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif mode == COLOUR_MODE_1BIT:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(
                grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 2. Saturation: scale the HSV S channel. Meaningless in
        # greyscale/1-bit modes, so skip it there.
        saturation = float(filters.get("saturation", IMAGE_SATURATION))
        if mode == COLOUR_MODE_COLOUR and abs(saturation - 1.0) > 0.001:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=saturation)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 3. Contrast: linear alpha scaling.
        contrast = float(filters.get("contrast", IMAGE_CONTRAST))
        if abs(contrast - 1.0) > 0.001:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)

        # 4. Median blur: cheap salt-and-pepper noise suppression.
        # Slider value N maps to an odd kernel (2N+1); 0 disables.
        level = int(filters.get("median_blur", IMAGE_MEDIAN_BLUR))
        if level > 0:
            frame = cv2.medianBlur(frame, 2 * level + 1)

        # 5. Sharpness: unsharp mask (blur, subtract, add back).
        sharpness = float(filters.get("sharpness", IMAGE_SHARPNESS))
        if abs(sharpness) > 0.001:
            blurred = cv2.GaussianBlur(
                frame, (0, 0), IMAGE_UNSHARP_SIGMA,
            )
            frame = cv2.addWeighted(
                frame, 1.0 + sharpness, blurred, -sharpness, 0,
            )

        return frame
