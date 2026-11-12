"""Abstract base for decoders."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    """A single decoded detection result.

    Passed from decoders to the results manager.
    """
    type: str                     # "QR", "EAN-13", "CODE128", "TXT"
    value: str                    # Decoded content
    confidence: float             # 0.0 to 1.0 (or 0-100 normalised)
    bbox: Optional[tuple] = None  # (x, y, w, h)
    points: Optional[list] = None  # Polygon points (for zbar)


class Decoder:
    """Abstract base class for frame decoders."""

    def decode(self, frame) -> list[Detection]:
        """Decode a frame, returning a list of Detection objects."""
        raise NotImplementedError