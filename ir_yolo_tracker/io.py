"""Frame loading helpers for infrared data."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import validate_ir_frame


def list_pickle_frames(data_dir: str | Path, pattern: str = "*.pickle") -> list[Path]:
    """Return sorted pickle frame paths from ``data_dir``."""

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    frames = sorted(root.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No pickle frames matching {pattern!r} found in: {root}")
    return frames


def extract_frame_array(payload: Any) -> np.ndarray:
    """Extract a numpy infrared frame from a pickle payload.

    The project data stores the frame directly as ``numpy.ndarray``. Dict/list
    support is provided for callers whose capture pipeline wraps the array.
    """

    if isinstance(payload, np.ndarray):
        return payload

    if isinstance(payload, dict):
        preferred_keys = ("frame", "image", "infrared", "ir", "data", "array")
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, np.ndarray):
                return value
        for value in payload.values():
            if isinstance(value, np.ndarray):
                return value

    if isinstance(payload, (list, tuple)):
        for value in payload:
            if isinstance(value, np.ndarray):
                return value

    raise TypeError(f"Cannot find a numpy.ndarray infrared frame in payload {type(payload)!r}.")


def load_pickle_frame(path: str | Path) -> np.ndarray:
    """Load and validate one two-dimensional ``uint16`` infrared pickle frame."""

    frame_path = Path(path)
    with frame_path.open("rb") as file:
        payload = pickle.load(file)

    frame = extract_frame_array(payload)
    validate_ir_frame(frame)
    return frame


def iter_pickle_frames(
    data_dir: str | Path,
    pattern: str = "*.pickle",
) -> Iterator[tuple[Path, np.ndarray]]:
    """Yield ``(path, frame)`` pairs from a pickle frame directory."""

    for path in list_pickle_frames(data_dir, pattern=pattern):
        yield path, load_pickle_frame(path)


def preload_pickle_frames(
    data_dir: str | Path,
    pattern: str = "*.pickle",
    *,
    progress: bool = False,
    progress_desc: str = "Preloading infrared frames",
) -> list[tuple[Path, np.ndarray]]:
    """Load all pickle frames into memory before playback or batch processing."""

    paths = list_pickle_frames(data_dir, pattern=pattern)
    if progress:
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise ImportError(
                "tqdm is required for preload progress bars. "
                "Install the project dependencies with `python -m pip install -e .`."
            ) from exc

        iterator = tqdm(paths, desc=progress_desc, unit="frame")
    else:
        iterator = paths

    return [(path, load_pickle_frame(path)) for path in iterator]
