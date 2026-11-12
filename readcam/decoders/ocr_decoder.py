"""OCR text decoder using Tesseract."""

import cv2
import numpy as np

from .base import Decoder, Detection
from ..config import TEXT_MIN_CONFIDENCE, TEXT_MIN_LENGTH
from ..detector import find_text_regions


class OCRDecoder(Decoder):
    """Decodes text regions in a frame using Tesseract OCR.

    Slower than zbar -- designed to run on a background thread
    at a lower frequency (e.g. every 1-2 seconds).
    """

    def __init__(
        self,
        min_confidence: int = TEXT_MIN_CONFIDENCE,
        min_length: int = TEXT_MIN_LENGTH,
        debug: bool = False,
    ):
        self.min_confidence = min_confidence
        self.min_length = min_length
        self.debug = debug
        self._available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, OSError, EnvironmentError):
            return False

    @property
    def available(self) -> bool:
        return self._available

    def decode_frame(self, frame: np.ndarray) -> list[Detection]:
        """Find text regions in *frame* and decode each with Tesseract.

        Returns a list of Detection objects (may be empty).
        """
        if not self._available:
            return []

        import pytesseract

        regions = find_text_regions(frame)
        results = []

        for (x, y, w, h) in regions:
            # Extract region of interest with a small padding
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            try:
                data = pytesseract.image_to_data(
                    roi,
                    output_type=pytesseract.Output.DICT,
                    config="--psm 6 --oem 3",
                )

                text_parts: list[str] = []
                confidences: list[int] = []

                for i, text in enumerate(data.get("text", [])):
                    conf = int(data.get("conf", [0])[i])
                    text = text.strip()
                    if text and conf > self.min_confidence:
                        text_parts.append(text)
                        confidences.append(conf)

                if not text_parts:
                    continue

                full_text = " ".join(text_parts)
                if len(full_text) < self.min_length:
                    continue

                avg_confidence = sum(confidences) / len(confidences) / 100.0

                results.append(Detection(
                    type="TXT",
                    value=full_text,
                    confidence=avg_confidence,
                    bbox=(x, y, w, h),
                ))

            except Exception as exc:
                if self.debug:
                    import logging
                    logging.debug("OCR error at %s: %s", (x, y, w, h), exc)
                continue

        return results