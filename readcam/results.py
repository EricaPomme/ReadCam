"""Results manager with confidence accumulation and decay."""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .config import (
    CONFIDENCE_MAX,
    CONFIDENCE_DECAY_DELAY,
    CONFIDENCE_REMOVE_DELAY,
    TEXT_REQUIRED_HITS,
)

logger = logging.getLogger(__name__)


@dataclass
class Result:
    """A single detection result tracked by the manager."""

    type: str                     # "QR", "EAN-13", "CODE128", "TXT"
    value: str                    # Decoded content
    confidence: int = 1           # Number of consecutive detections
    first_seen: float = 0.0       # Unix timestamp of first detection
    last_seen: float = 0.0        # Unix timestamp of most recent detection
    bbox: Optional[tuple] = None  # (x, y, w, h) most recent bounding box

    @property
    def key(self) -> tuple:
        """Unique key for deduplication."""
        return (self.type, self.value)


class ResultsManager:
    """Manages accumulation, deduplication, and decay of detection results.

    Each unique (type, value) pair is tracked once. Confidence
    increments on repeated sightings and decays when a result goes
    unseen. Results with zero confidence are eventually removed.
    """

    def __init__(self):
        self._results: dict[tuple[str, str], Result] = {}
        self._lock = threading.Lock()

    def add_or_update(
        self,
        det_type: str,
        value: str,
        bbox: Optional[tuple] = None,
    ) -> Optional[Result]:
        """Add or update a non-text detection.

        Returns the Result if added/updated, or None for text
        results that haven't reached the required hit threshold.
        """
        key = (det_type, value)
        now = time.time()

        with self._lock:
            if key in self._results:
                result = self._results[key]
                result.confidence = min(result.confidence + 1, CONFIDENCE_MAX)
                result.last_seen = now
                if bbox:
                    result.bbox = bbox
                return result

            # Text results go through add_text_hit instead
            if det_type == "TXT":
                return None

            result = Result(
                type=det_type,
                value=value,
                confidence=1,
                first_seen=now,
                last_seen=now,
                bbox=bbox,
            )
            self._results[key] = result
            logger.info("New result: %s %r", det_type, value)
            return result

    def add_text_hit(
        self,
        value: str,
        bbox: Optional[tuple] = None,
    ) -> Optional[Result]:
        """Register a text detection sighting.

        Text results require multiple consistent detections before
        they appear (to reduce OCR false positives). Returns the
        Result once it passes the threshold, or None otherwise.
        """
        key = ("TXT", value)
        now = time.time()

        with self._lock:
            if key in self._results:
                result = self._results[key]
                result.confidence = min(result.confidence + 1, CONFIDENCE_MAX)
                result.last_seen = now
                if bbox:
                    result.bbox = bbox
                return result if result.confidence >= TEXT_REQUIRED_HITS else None

            result = Result(
                type="TXT",
                value=value,
                confidence=1,
                first_seen=now,
                last_seen=now,
                bbox=bbox,
            )
            self._results[key] = result
            logger.debug(
                "Text sighting %d/%d: %r",
                result.confidence, TEXT_REQUIRED_HITS, value,
            )
            return None  # Requires another sighting to appear

    def remove(self, det_type: str, value: str) -> bool:
        """Remove a result by type and value."""
        key = (det_type, value)
        with self._lock:
            if key in self._results:
                del self._results[key]
                return True
            return False

    def decay(self) -> None:
        """Decay stale results and purge very old ones.

        Results unseen for CONFIDENCE_DECAY_DELAY seconds start
        losing confidence. Results unseen for CONFIDENCE_REMOVE_DELAY
        seconds are purged entirely.
        """
        now = time.time()
        to_remove = []

        with self._lock:
            for key, result in self._results.items():
                elapsed = now - result.last_seen
                if elapsed > CONFIDENCE_REMOVE_DELAY:
                    to_remove.append(key)
                elif elapsed > CONFIDENCE_DECAY_DELAY:
                    decay_amount = int((elapsed - CONFIDENCE_DECAY_DELAY) / 2) + 1
                    result.confidence = max(0, result.confidence - decay_amount)

            for key in to_remove:
                del self._results[key]

        for key in to_remove:
            logger.info("Removed stale result: %s %r", key[0], key[1])

    def get_sorted(self) -> list[Result]:
        """Return all results, sorted by confidence then recency."""
        with self._lock:
            results = list(self._results.values())

        results.sort(
            key=lambda r: (r.confidence, r.last_seen),
            reverse=True,
        )
        return results

    def get_active(self) -> list[Result]:
        """Return results with confidence > 0 (currently visible)."""
        return [r for r in self.get_sorted() if r.confidence > 0]

    def clear(self) -> None:
        """Remove all results."""
        with self._lock:
            self._results.clear()

    @property
    def count(self) -> int:
        """Number of unique results being tracked."""
        with self._lock:
            return len(self._results)