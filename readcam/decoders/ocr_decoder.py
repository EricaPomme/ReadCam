"""OCR text decoder using Tesseract."""

from typing import Optional

import cv2
import numpy as np

from .base import Detection, Decoder
from ..config import (
    OCR_MAX_WIDTH,
    TEXT_MIN_ALNUM_RATIO,
    TEXT_MIN_CONFIDENCE,
    TEXT_MIN_LETTERS,
    TEXT_MIN_LENGTH,
)
from ..install import configure_tesseract


def looks_like_text(
    value: str,
    min_length: int = TEXT_MIN_LENGTH,
    min_alnum_ratio: float = TEXT_MIN_ALNUM_RATIO,
    min_letters: int = TEXT_MIN_LETTERS,
) -> bool:
    """Heuristic filter to reject OCR pareidolia (text in noise).

    Requires the string to be long enough, mostly alphanumeric,
    and to contain a minimum number of letters. Junk like 'ae',
    'i>.', 'ik', or symbols-only strings fail. Thresholds may be
    overridden live by the UI via the optional parameters.
    """
    value = value.strip()
    if len(value) < min_length:
        return False

    alnum = sum(1 for c in value if c.isalnum())
    if alnum / len(value) < min_alnum_ratio:
        return False

    letters = sum(1 for c in value if c.isalpha())
    if letters < min_letters:
        return False

    return True


class OCRDecoder(Decoder):
    """Decodes text in a frame using Tesseract OCR.

    Runs word-level OCR over the whole frame (sparse-text mode),
    then groups words into lines. Slower than zbar -- designed to
    run on a background thread at a lower frequency (e.g. 1-2 s).
    """

    def __init__(
        self,
        min_confidence: int = TEXT_MIN_CONFIDENCE,
        min_length: int = TEXT_MIN_LENGTH,
        filter_holder: Optional[dict] = None,
        debug: bool = False,
    ):
        """Create the decoder.

        *filter_holder* is an optional shared dict of live filter
        overrides written by the UI sliders; missing keys fall back to
        the constructor defaults (themselves sourced from config).
        """
        self.min_confidence = min_confidence
        self.min_length = min_length
        self._filters = filter_holder if filter_holder is not None else {}
        self.debug = debug
        self._available = self._check_available()

    def _get(self, key: str, fallback):
        """Read a live filter override, falling back to the default."""
        return self._filters.get(key, fallback)

    @staticmethod
    def _check_available() -> bool:
        try:
            import pytesseract
        except ImportError:
            return False

        # Try PATH first; fall back to discovered install locations
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            pass

        try:
            from ..install import configure_tesseract
            return configure_tesseract()
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def decode(self, frame: np.ndarray) -> list[Detection]:
        """Run OCR over *frame*, returning line-grouped text detections.

        Returns a list of Detection objects (may be empty).
        """
        if not self._available:
            return []

        import pytesseract

        try:
            gray = _to_gray(frame)

            # Downscale large frames -- OCR cost scales with pixel count.
            # Bounding boxes are scaled back to original frame coordinates.
            scale = 1.0
            h, w = gray.shape[:2]
            if w > OCR_MAX_WIDTH:
                scale = OCR_MAX_WIDTH / w
                gray = cv2.resize(
                    gray, (OCR_MAX_WIDTH, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            data = pytesseract.image_to_data(
                gray,
                output_type=pytesseract.Output.DICT,
                config="--psm 11 --oem 3",
            )
        except Exception as exc:
            if self.debug:
                import logging
                logging.debug("OCR error: %s", exc)
            return []

        # Group words into lines keyed by (block, par, line)
        min_confidence = self._get("min_confidence", self.min_confidence)
        min_length = self._get("min_length", self.min_length)
        min_alnum_ratio = self._get("min_alnum_ratio", TEXT_MIN_ALNUM_RATIO)
        min_letters = self._get("min_letters", TEXT_MIN_LETTERS)

        lines: dict[tuple, dict] = {}
        texts = data.get("text", [])
        confs = data.get("conf", [])
        blocks = data.get("block_num", [])
        paras = data.get("par_num", [])
        line_nums = data.get("line_num", [])
        lefts = data.get("left", [])
        tops = data.get("top", [])
        widths = data.get("width", [])
        heights = data.get("height", [])

        for i, word in enumerate(texts):
            word = str(word).strip()
            try:
                conf = int(float(confs[i]))
            except (ValueError, TypeError):
                conf = -1

            if not word or conf < min_confidence:
                continue

            key = (blocks[i], paras[i], line_nums[i])
            entry = lines.setdefault(key, {"words": [], "conf": [], "box": None})

            entry["words"].append(word)
            entry["conf"].append(conf)

            x, y, w, h = lefts[i], tops[i], widths[i], heights[i]
            if entry["box"] is None:
                entry["box"] = (x, y, w, h)
            else:
                bx, by, bw, bh = entry["box"]
                nx = min(bx, x)
                ny = min(by, y)
                nw = max(bx + bw, x + w) - nx
                nh = max(by + bh, y + h) - ny
                entry["box"] = (nx, ny, nw, nh)

        results: list[Detection] = []
        for entry in lines.values():
            text = " ".join(entry["words"])
            if len(text) < min_length:
                continue
            if not looks_like_text(
                text,
                min_length=min_length,
                min_alnum_ratio=min_alnum_ratio,
                min_letters=min_letters,
            ):
                continue

            avg_conf = sum(entry["conf"]) / len(entry["conf"]) / 100.0
            bbox = entry["box"]
            if scale != 1.0 and bbox is not None:
                x, y, bw, bh = bbox
                bbox = (
                    int(x / scale), int(y / scale),
                    int(bw / scale), int(bh / scale),
                )
            results.append(Detection(
                type="TXT",
                value=text,
                confidence=avg_conf,
                bbox=bbox,
            ))

        return results


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR (or already-grayscale) frame to grayscale."""
    import cv2
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
