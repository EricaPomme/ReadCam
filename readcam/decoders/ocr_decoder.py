"""OCR text decoder using Tesseract."""

import cv2
import numpy as np

from .base import Detection, Decoder
from ..config import OCR_MAX_WIDTH, TEXT_MIN_CONFIDENCE, TEXT_MIN_LENGTH
from ..install import configure_tesseract


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

            if not word or conf < self.min_confidence:
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
            if len(text) < self.min_length:
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
