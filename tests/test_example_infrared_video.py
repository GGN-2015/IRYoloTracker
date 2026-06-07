from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from example_infrared_video import (
    PreviewMarkerDetector,
    extract_frame_array,
    list_pickle_frames,
    load_pickle_frame,
    resolve_weights,
)


def test_extract_frame_array_accepts_direct_array() -> None:
    frame = np.zeros((512, 512), dtype=np.uint16)

    assert extract_frame_array(frame) is frame


def test_extract_frame_array_accepts_common_dict_keys() -> None:
    frame = np.zeros((512, 512), dtype=np.uint16)

    assert extract_frame_array({"image": frame}) is frame


def test_load_pickle_frame_reads_uint16_array(tmp_path: Path) -> None:
    frame = np.zeros((512, 512), dtype=np.uint16)
    path = tmp_path / "0000001.pickle"
    with path.open("wb") as file:
        pickle.dump(frame, file)

    loaded = load_pickle_frame(path)

    assert loaded.shape == (512, 512)
    assert loaded.dtype == np.uint16


def test_list_pickle_frames_sorts_by_name(tmp_path: Path) -> None:
    (tmp_path / "0000002.pickle").write_bytes(b"")
    (tmp_path / "0000001.pickle").write_bytes(b"")

    frames = list_pickle_frames(tmp_path)

    assert [path.name for path in frames] == ["0000001.pickle", "0000002.pickle"]


def test_resolve_weights_returns_bundled_weights_when_run_weights_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    weights = resolve_weights(None)

    assert weights is not None
    assert weights.exists()


def test_preview_marker_detector_finds_bright_circle() -> None:
    frame = np.zeros((512, 512), dtype=np.uint16)
    yy, xx = np.ogrid[:512, :512]
    mask = (xx - 250) ** 2 + (yy - 200) ** 2 <= 8**2
    frame[mask] = 60_000

    detections = PreviewMarkerDetector().detect(frame)

    assert len(detections) == 1
    x_min, y_min, x_max, y_max = detections[0].bbox_xyxy
    assert 240 <= x_min <= 252
    assert 192 <= y_min <= 204
    assert 248 <= x_max <= 260
    assert 200 <= y_max <= 212
    assert detections[0].confidence > 0.5
