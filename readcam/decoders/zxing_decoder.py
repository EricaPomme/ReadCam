"""Data Matrix and general barcode decoder using zxing-cpp.

Complements zbar: zxing-cpp is a modern, wheel-distributed decoder
(no system DLLs) whose Data Matrix support is much stronger than
zbar's for small or low-resolution codes.
"""

import cv2
import numpy as np

from .base import Detection, Decoder


class ZXingDecoder(Decoder):
    """Decodes 1D/2D barcodes using the zxing-cpp library."""

    # Map zxing-cpp format names to our canonical categories
    TYPE_MAP = {
        "DataMatrix": "DM",
        "QRCode": "QR",
        "EAN13": "EAN-13",
        "EAN8": "EAN-8",
        "UPCA": "UPC-A",
        "UPCE": "UPC-E",
        "Code128": "CODE128",
        "Code39": "CODE39",
        "Code93": "CODE93",
        "Codabar": "CODABAR",
        "ITF": "I2OF5",
        "PDF417": "PDF417",
        "DataBarLimited": "DATABAR",
        "DataBarExpanded": "DATABAR",
        "DataBar": "DATABAR",
        "Aztec": "AZTEC",
    }

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            import zxingcpp  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

    @property
    def available(self) -> bool:
        return self._available

    def decode(self, frame: np.ndarray) -> list[Detection]:
        """Run zxing-cpp decode on an OpenCV BGR frame."""
        if not self._available:
            return []

        import zxingcpp

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            symbols = zxingcpp.read_barcodes(gray)
        except Exception as exc:
            if self.debug:
                import logging
                logging.debug("zxing decode error: %s", exc)
            return []

        results: list[Detection] = []
        for sym in symbols:
            fmt = sym.format.name
            our_type = self.TYPE_MAP.get(fmt, fmt.upper())

            # Position gives the four corner points
            pos = sym.position
            xs = [pos.top_left.x, pos.top_right.x,
                  pos.bottom_left.x, pos.bottom_right.x]
            ys = [pos.top_left.y, pos.top_right.y,
                  pos.bottom_left.y, pos.bottom_right.y]
            bbox = (
                int(min(xs)), int(min(ys)),
                int(max(xs) - min(xs)), int(max(ys) - min(ys)),
            )

            results.append(Detection(
                type=our_type,
                value=sym.text,
                confidence=1.0,
                bbox=bbox,
            ))

        return results