"""Preprocessing for single-channel uint16 infrared frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FrameShape = tuple[int, int]
ModelInputChannels = Literal[1, 3]


@dataclass(frozen=True)
class PreprocessConfig:
    """Intensity normalization settings for infrared frames.

    The input frame remains single-channel. When ``model_input_channels`` is 3,
    the normalized gray image is copied into three identical channels only to
    satisfy standard YOLO model input shapes.
    """

    frame_shape: FrameShape | None = None
    lower_percentile: float = 0.1
    upper_percentile: float = 99.9

    def __post_init__(self) -> None:
        if self.frame_shape is not None:
            if len(self.frame_shape) != 2:
                raise ValueError("frame_shape must contain exactly two dimensions.")
            if self.frame_shape[0] <= 0 or self.frame_shape[1] <= 0:
                raise ValueError("frame_shape dimensions must be positive.")
        if not 0.0 <= self.lower_percentile < self.upper_percentile <= 100.0:
            raise ValueError(
                "Percentiles must satisfy "
                "0 <= lower_percentile < upper_percentile <= 100."
            )


def validate_ir_frame(frame: np.ndarray, frame_shape: FrameShape | None = None) -> None:
    """Validate that ``frame`` is a single-channel uint16 infrared image."""

    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray.")
    if frame.dtype != np.uint16:
        raise TypeError("frame must have dtype numpy.uint16.")
    if frame.ndim != 2:
        raise ValueError(f"frame must be a two-dimensional grayscale image, got shape {frame.shape}.")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError(f"frame dimensions must be positive, got shape {frame.shape}.")
    if frame_shape is not None and frame.shape != frame_shape:
        raise ValueError(f"frame must have shape {frame_shape}, got {frame.shape}.")


def normalize_uint16_to_uint8(
    frame: np.ndarray,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Map a uint16 infrared frame to uint8 using robust percentile clipping."""

    config = config or PreprocessConfig()
    validate_ir_frame(frame, config.frame_shape)

    lower, upper = np.percentile(
        frame,
        [config.lower_percentile, config.upper_percentile],
    )

    if upper <= lower:
        lower = float(frame.min())
        upper = float(frame.max())

    if upper <= lower:
        return np.zeros(frame.shape, dtype=np.uint8)

    normalized = (frame.astype(np.float32) - float(lower)) * (255.0 / float(upper - lower))
    return np.clip(normalized, 0.0, 255.0).astype(np.uint8)


def prepare_yolo_image(
    frame: np.ndarray,
    config: PreprocessConfig | None = None,
    model_input_channels: ModelInputChannels = 3,
) -> np.ndarray:
    """Prepare a single infrared frame for YOLO inference.

    Args:
        frame: A single-channel two-dimensional ``numpy.uint16`` infrared frame.
        config: Optional preprocessing settings.
        model_input_channels: Use ``3`` for standard YOLO weights, or ``1`` for
            a YOLO model trained with a one-channel input layer.

    Returns:
        A contiguous uint8 array shaped ``(H, W, C)`` where ``C`` is 1 or 3.
    """

    if model_input_channels not in (1, 3):
        raise ValueError("model_input_channels must be 1 or 3.")

    gray = normalize_uint16_to_uint8(frame, config)
    image = gray[:, :, None]

    if model_input_channels == 3:
        image = np.repeat(image, 3, axis=2)

    return np.ascontiguousarray(image)
