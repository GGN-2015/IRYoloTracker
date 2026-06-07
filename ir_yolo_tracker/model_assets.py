"""Bundled model asset helpers."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

DEFAULT_MODEL_NAME = "pseudo_ir_marker_ball_yolo11n.pt"


def get_default_model_path() -> Path:
    """Return the bundled bootstrap YOLO model path."""

    return Path(str(resources.files("ir_yolo_tracker.models").joinpath(DEFAULT_MODEL_NAME)))
