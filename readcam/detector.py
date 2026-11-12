"""Region detection via image processing to find text-like areas."""

import cv2
import numpy as np

from .config import (
    CONTOUR_MIN_AREA,
    CONTOUR_MAX_AREA,
    CONTOUR_MIN_ASPECT_RATIO,
    CONTOUR_MAX_ASPECT_RATIO,
)


def find_text_regions(
    frame: np.ndarray,
    min_area: int = CONTOUR_MIN_AREA,
    max_area: int = CONTOUR_MAX_AREA,
    min_aspect: float = CONTOUR_MIN_ASPECT_RATIO,
    max_aspect: float = CONTOUR_MAX_ASPECT_RATIO,
) -> list[tuple[int, int, int, int]]:
    """Find regions in the frame that likely contain text.

    Uses thresholding and contour detection to locate candidate
    text areas, then filters by size and aspect ratio.

    Returns:
        List of (x, y, w, h) bounding boxes for each text region.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold via Otsu's method
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    # Find external contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        if w > 0 and h > 0:
            aspect = w / h
            if aspect < min_aspect or aspect > max_aspect:
                continue

        regions.append((x, y, w, h))

    return _merge_boxes(regions)


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]],
    overlap_threshold: float = 0.3,
    distance_threshold: int = 20,
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping or nearby bounding boxes."""
    if not boxes:
        return []

    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                if _should_merge(boxes[i], boxes[j], overlap_threshold, distance_threshold):
                    boxes[i] = _union(boxes[i], boxes[j])
                    boxes.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1

    return boxes


def _should_merge(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    overlap_threshold: float,
    distance_threshold: int,
) -> bool:
    """Check whether two bounding boxes should be merged."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    # Intersection
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    if ix < ix2 and iy < iy2:
        intersection = (ix2 - ix) * (iy2 - iy)
        area_a = aw * ah
        area_b = bw * bh
        overlap = intersection / min(area_a, area_b) if min(area_a, area_b) > 0 else 0
        if overlap > overlap_threshold:
            return True

    # Proximity based on centre distance
    cx_a, cy_a = ax + aw // 2, ay + ah // 2
    cx_b, cy_b = bx + bw // 2, by + bh // 2
    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    if dist < distance_threshold:
        return True

    return False


def _union(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the union bounding box of two rectangles."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x = min(ax, bx)
    y = min(ay, by)
    w = max(ax + aw, bx + bw) - x
    h = max(ay + ah, by + bh) - y
    return (x, y, w, h)