"""Barcode and QR code decoder wrapping zbar."""

import cv2
import numpy as np

from .base import Decoder, Detection


class ZBarDecoder(Decoder):
    """Decodes QR codes and linear barcodes using the zbar library.

    Extremely fast -- suitable for real-time frame-by-frame decode.
    """

    # Map zbar symbology names to our canonical categories
    # (keys match pyzbar's ZBarSymbol enum member names)
    TYPE_MAP = {
        "QRCODE": "QR",
        "EAN-13": "EAN-13",
        "EAN13": "EAN-13",
        "EAN-8": "EAN-8",
        "EAN8": "EAN-8",
        "UPC-A": "UPC-A",
        "UPCA": "UPC-A",
        "UPC-E": "UPC-E",
        "UPCE": "UPC-E",
        "CODE-128": "CODE128",
        "CODE128": "CODE128",
        "CODE-39": "CODE39",
        "CODE39": "CODE39",
        "CODE-93": "CODE93",
        "CODE93": "CODE93",
        "CODABAR": "CODABAR",
        "I2/5": "I2OF5",
        "I25": "I2OF5",
        "PDF417": "PDF417",
        "DATABAR": "DATABAR",
        "DATABAR_EXP": "DATABAR",
    }

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            from pyzbar.pyzbar import decode  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

    @property
    def available(self) -> bool:
        return self._available

    def decode(self, frame: np.ndarray) -> list[Detection]:
        """Run zbar decode on an OpenCV BGR frame.

        Returns a list of Detection objects (may be empty).
        """
        if not self._available:
            return []

        from pyzbar.pyzbar import decode

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            symbols = decode(gray)
        except Exception as exc:
            if self.debug:
                import logging
                logging.debug("zbar decode error: %s", exc)
            return []

        results = []
        for sym in symbols:
            raw_data = sym.data
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8", errors="replace")
            else:
                raw_data = str(raw_data)

            sym_type = getattr(sym, 'type', 'UNKNOWN')
            # pyzbar returns a ZBarSymbol IntEnum; str() of it is the
            # numeric value on Python 3.11+, so use .name instead.
            sym_name = str(getattr(sym_type, 'name', sym_type)).upper()
            our_type = self.TYPE_MAP.get(sym_name, sym_name)

            # Polygon points -> bounding box
            points = getattr(sym, 'points', [])
            bbox = None
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

            results.append(Detection(
                type=our_type,
                value=raw_data,
                confidence=1.0,  # zbar decode is binary
                bbox=bbox,
                points=points,
            ))

        return results