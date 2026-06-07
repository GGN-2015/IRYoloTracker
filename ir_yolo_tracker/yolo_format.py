"""Helpers for YOLO detection labels."""

from __future__ import annotations

from pathlib import Path


def xyxy_to_yolo(
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert an ``xyxy`` box in pixels to normalized YOLO ``xywh``."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive.")

    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    x_min = min(max(x_min, 0.0), float(image_width))
    x_max = min(max(x_max, 0.0), float(image_width))
    y_min = min(max(y_min, 0.0), float(image_height))
    y_max = min(max(y_max, 0.0), float(image_height))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"invalid xyxy box: {bbox_xyxy}")

    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + box_width / 2.0
    y_center = y_min + box_height / 2.0

    return (
        x_center / float(image_width),
        y_center / float(image_height),
        box_width / float(image_width),
        box_height / float(image_height),
    )


def format_yolo_label_line(
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> str:
    """Format one single-class YOLO label row."""

    x_center, y_center, width, height = xyxy_to_yolo(
        bbox_xyxy,
        image_width=image_width,
        image_height=image_height,
    )
    return (
        f"{class_id} "
        f"{x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
    )


def write_yolo_label_file(
    path: str | Path,
    boxes_xyxy: list[tuple[float, float, float, float]] | list[list[float]],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> None:
    """Write one YOLO label file for marker-ball boxes."""

    lines = [
        format_yolo_label_line(
            box,
            image_width=image_width,
            image_height=image_height,
            class_id=class_id,
        )
        for box in boxes_xyxy
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
