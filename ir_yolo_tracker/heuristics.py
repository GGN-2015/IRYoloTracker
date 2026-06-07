"""Non-YOLO marker-ball helpers used for bootstrapping and debugging."""

from __future__ import annotations

import cv2
import numpy as np

from .detections import MarkerDetection
from .preprocessing import normalize_uint16_to_uint8, validate_ir_frame


class BrightCircleDetector:
    """Detect bright circular blobs without a neural model.

    This detector is useful for pseudo-label bootstrapping and visual debugging.
    The production detector remains ``IRMarkerTracker``.
    """

    mode_name = "bright-circle"

    def __init__(
        self,
        *,
        min_area: int = 8,
        max_area: int = 2_500,
        min_circularity: float = 0.45,
        threshold_percentile: float = 99.7,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.threshold_percentile = threshold_percentile

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        """Return bright circular marker candidates."""

        validate_ir_frame(frame)
        gray = normalize_uint16_to_uint8(frame)
        threshold = max(1, int(np.percentile(gray, self.threshold_percentile)))
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[MarkerDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = float(cv2.arcLength(contour, closed=True))
            if perimeter <= 0.0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            aspect_ratio = width / max(height, 1)
            if not 0.55 <= aspect_ratio <= 1.8:
                continue

            roi = gray[y : y + height, x : x + width]
            confidence = min(1.0, max(0.0, circularity) * (float(roi.max()) / 255.0))
            detections.append(
                MarkerDetection(
                    bbox_xyxy=(float(x), float(y), float(x + width), float(y + height)),
                    confidence=confidence,
                    class_id=0,
                    class_name="ir_marker_ball",
                )
            )

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections
