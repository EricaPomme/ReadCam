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
    confidence: float = 1.0       # ~seconds visible (capped at CONFIDENCE_MAX)
    hits: int = 1                 # Number of detections
    first_seen: float = 0.0       # Unix timestamp of first detection
    last_seen: float = 0.0        # Unix timestamp of most recent detection
    bbox: Optional[tuple] = None  # (x, y, w, h) most recent bounding box

    @property
    def visible_seconds(self) -> float:
        """Total time this result has been visible."""
        if self.last_seen and self.first_seen:
            return self.last_seen - self.first_seen
        return 0.0

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

    def __init__(self, filter_holder: Optional[dict] = None):
        """Create the manager.

        *filter_holder* is an optional shared dict of live filter
        overrides (``required_hits``, ``decay_delay``, ``remove_delay``)
        written by the UI; missing keys fall back to config defaults.
        """
        self._results: dict[tuple[str, str], Result] = {}
        self._filters = filter_holder if filter_holder is not None else {}
        self._lock = threading.Lock()
        self._version = 0  # bumped on every mutation (UI refresh throttling)

    @property
    def version(self) -> int:
        """Monotonic change counter for cheap dirty-checking."""
        return self._version

    def add_or_update(
        self,
        det_type: str,
        value: str,
        bbox: Optional[tuple] = None,
    ) -> Optional[Result]:
        """Add or update a non-text detection.

        Confidence accumulates as roughly *seconds visible* rather than
        hit count, so slow decoders (OCR, ~1 hit per 1.5 s) and fast
        decoders (zbar, ~10 hits per second) rank fairly.
        Returns the Result if added/updated, or None for text
        results that haven't reached the required hit threshold.
        """
        key = (det_type, value)
        now = time.time()

        with self._lock:
            self._version += 1
            if key in self._results:
                result = self._results[key]
                result.confidence = self._accumulate(result, now)
                result.hits += 1
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
                confidence=1.0,
                hits=1,
                first_seen=now,
                last_seen=now,
                bbox=bbox,
            )
            self._results[key] = result
            logger.info("New result: %s %r", det_type, value)
            return result

    @staticmethod
    def _accumulate(result: Result, now: float) -> float:
        """Grow confidence by elapsed time since the last sighting."""
        elapsed = now - result.last_seen
        # Cap the delta so a single long gap doesn't rocket the score
        delta = min(max(elapsed, 0.0), 5.0)
        return min(result.confidence + delta, CONFIDENCE_MAX)

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
            self._version += 1
            if key in self._results:
                result = self._results[key]
                result.confidence = self._accumulate(result, now)
                result.hits += 1
                result.last_seen = now
                if bbox:
                    result.bbox = bbox
                required_hits = self._filters.get(
                    "required_hits", TEXT_REQUIRED_HITS,
                )
                return result if result.hits >= required_hits else None

            result = Result(
                type="TXT",
                value=value,
                confidence=1.0,
                hits=1,
                first_seen=now,
                last_seen=now,
                bbox=bbox,
            )
            self._results[key] = result
            logger.debug(
                "Text sighting %d/%d: %r",
                result.hits,
                self._filters.get("required_hits", TEXT_REQUIRED_HITS),
                value,
            )
            return None  # Requires another sighting to appear

    def remove(self, det_type: str, value: str) -> bool:
        """Remove a result by type and value."""
        key = (det_type, value)
        with self._lock:
            self._version += 1
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
        decay_delay = self._filters.get("decay_delay", CONFIDENCE_DECAY_DELAY)
        remove_delay = self._filters.get("remove_delay", CONFIDENCE_REMOVE_DELAY)

        with self._lock:
            for key, result in self._results.items():
                elapsed = now - result.last_seen
                if elapsed > remove_delay:
                    to_remove.append(key)
                elif elapsed > decay_delay:
                    decay_amount = int((elapsed - decay_delay) / 2) + 1
                    result.confidence = max(0, result.confidence - decay_amount)

            if to_remove:
                self._version += 1
                for key in to_remove:
                    del self._results[key]

        for key in to_remove:
            logger.info("Removed stale result: %s %r", key[0], key[1])

    def get_sorted(self) -> list[Result]:
        """Return all results ranked by prominence.

        Primary: confidence (how consistently detected).
        Secondary: total visible time (how long it has been in view).
        Long-standing results therefore stay high in the list.
        """
        with self._lock:
            results = list(self._results.values())

        results.sort(
            key=lambda r: (r.confidence, r.visible_seconds, r.last_seen),
            reverse=True,
        )
        return results

    def get_active(self) -> list[Result]:
        """Return results with confidence > 0 (currently visible)."""
        return [r for r in self.get_sorted() if r.confidence > 0]

    def clear(self) -> None:
        """Remove all results."""
        with self._lock:
            self._version += 1
            self._results.clear()

    @property
    def count(self) -> int:
        """Number of unique results being tracked."""
        with self._lock:
            return len(self._results)