"""Detection result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarkerDetection:
    """One infrared marker ball detection.

    Coordinates use YOLO-style ``xyxy`` image coordinates:
    ``(x_min, y_min, x_max, y_max)`` in pixels.
    """

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "ir_marker_ball"

    @property
    def center_xy(self) -> tuple[float, float]:
        """Return the center point of the bounding box in pixels."""

        x_min, y_min, x_max, y_max = self.bbox_xyxy
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center_xy": list(self.center_xy),
        }
