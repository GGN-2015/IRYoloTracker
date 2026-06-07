"""Convenience Python API for marker-ball detection."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .detections import MarkerDetection
from .io import iter_pickle_frames
from .preprocessing import ModelInputChannels, PreprocessConfig
from .tracker import IRMarkerTracker


def create_tracker(
    weights: str | Path | None = None,
    *,
    model: Any | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    image_size: int = 512,
    model_input_channels: ModelInputChannels = 3,
    preprocess_config: PreprocessConfig | None = None,
    device: str | int | None = None,
) -> IRMarkerTracker:
    """Create an ``IRMarkerTracker`` using the bundled model by default."""

    return IRMarkerTracker(
        weights,
        model=model,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        model_input_channels=model_input_channels,
        preprocess_config=preprocess_config,
        device=device,
    )


def detect_markers(
    frame: np.ndarray,
    *,
    tracker: IRMarkerTracker | None = None,
    weights: str | Path | None = None,
    model: Any | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    image_size: int = 512,
    model_input_channels: ModelInputChannels = 3,
    preprocess_config: PreprocessConfig | None = None,
    device: str | int | None = None,
) -> list[MarkerDetection]:
    """Detect marker balls in one two-dimensional ``uint16`` infrared frame.

    For repeated calls, create one ``IRMarkerTracker`` and pass it as
    ``tracker=...`` so the YOLO model is loaded only once.
    """

    active_tracker = tracker or create_tracker(
        weights,
        model=model,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        model_input_channels=model_input_channels,
        preprocess_config=preprocess_config,
        device=device,
    )
    return active_tracker.detect(frame)


def detect_marker_dicts(
    frame: np.ndarray,
    **kwargs: Any,
) -> list[dict[str, object]]:
    """Detect marker balls and return JSON-serializable dictionaries."""

    return [detection.to_dict() for detection in detect_markers(frame, **kwargs)]


def iter_marker_detections(
    frames: Iterable[np.ndarray],
    *,
    tracker: IRMarkerTracker | None = None,
    weights: str | Path | None = None,
    model: Any | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    image_size: int = 512,
    model_input_channels: ModelInputChannels = 3,
    preprocess_config: PreprocessConfig | None = None,
    device: str | int | None = None,
) -> Iterator[list[MarkerDetection]]:
    """Yield detections for an iterable of infrared frames."""

    active_tracker = tracker or create_tracker(
        weights,
        model=model,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        model_input_channels=model_input_channels,
        preprocess_config=preprocess_config,
        device=device,
    )
    for frame in frames:
        yield active_tracker.detect(frame)


def detect_marker_batch(
    frames: Iterable[np.ndarray],
    **kwargs: Any,
) -> list[list[MarkerDetection]]:
    """Detect marker balls in multiple frames."""

    return list(iter_marker_detections(frames, **kwargs))


def iter_pickle_detections(
    data_dir: str | Path,
    *,
    tracker: IRMarkerTracker | None = None,
    pattern: str = "*.pickle",
    weights: str | Path | None = None,
    model: Any | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    image_size: int = 512,
    model_input_channels: ModelInputChannels = 3,
    preprocess_config: PreprocessConfig | None = None,
    device: str | int | None = None,
) -> Iterator[tuple[Path, list[MarkerDetection]]]:
    """Yield ``(path, detections)`` for pickle frames in a directory."""

    active_tracker = tracker or create_tracker(
        weights,
        model=model,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        model_input_channels=model_input_channels,
        preprocess_config=preprocess_config,
        device=device,
    )
    for path, frame in iter_pickle_frames(data_dir, pattern=pattern):
        yield path, active_tracker.detect(frame)
