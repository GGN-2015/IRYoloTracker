"""Visualization helpers for marker-ball detections."""

from __future__ import annotations

import cv2
import numpy as np

from .detections import MarkerDetection
from .preprocessing import normalize_uint16_to_uint8


def draw_detections(
    frame: np.ndarray,
    detections: list[MarkerDetection],
    *,
    status: str | None = None,
    scale: float = 1.0,
    box_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    label_background: tuple[int, int, int] = (0, 80, 0),
) -> np.ndarray:
    """Draw detection boxes and confidences on an infrared frame.

    Args:
        frame: A two-dimensional ``uint16`` infrared frame, a grayscale uint8 image,
            or an existing BGR uint8 image.
        detections: Marker detections to overlay.
        status: Optional text drawn in the upper-left corner.
        scale: Optional display scaling factor.

    Returns:
        A BGR uint8 image suitable for OpenCV display or video writing.
    """

    canvas = _to_bgr_uint8(frame)

    for detection in detections:
        x_min, y_min, x_max, y_max = [int(round(value)) for value in detection.bbox_xyxy]
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), box_color, 2)
        _draw_label(
            canvas,
            f"{detection.confidence:.2f}",
            (x_min, max(16, y_min - 6)),
            background=label_background,
            foreground=text_color,
        )

    if status:
        _draw_label(
            canvas,
            status,
            (8, 20),
            background=(0, 0, 0),
            foreground=text_color,
        )

    return resize_for_display(canvas, scale)


def resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize an image by ``scale`` using nearest-neighbor interpolation."""

    if scale <= 0:
        raise ValueError("scale must be positive.")
    if scale == 1.0:
        return image
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def _to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2 and frame.dtype == np.uint16:
        gray = normalize_uint16_to_uint8(frame)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 2 and frame.dtype == np.uint8:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
        return frame.copy()
    raise TypeError(
        "frame must be a uint16 grayscale IR frame, a uint8 grayscale image, "
        "or a BGR uint8 image."
    )


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    background: tuple[int, int, int],
    foreground: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, min(x, image.shape[1] - width - 4))
    y = max(height + 4, min(y, image.shape[0] - baseline - 2))

    cv2.rectangle(
        image,
        (x - 2, y - height - 4),
        (x + width + 2, y + baseline + 2),
        background,
        thickness=-1,
    )
    cv2.putText(image, text, (x, y), font, font_scale, foreground, thickness, cv2.LINE_AA)
