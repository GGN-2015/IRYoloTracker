from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from example_infrared_video import bbox_iou
from example_infrared_video import fuse_pseudo_label_detections
from example_infrared_video import merge_marker_detections
from example_infrared_video import parse_args
from example_infrared_video import resolve_weights
from example_infrared_video import split_frame_paths
from ir_yolo_tracker import (
    BrightCircleDetector,
    MarkerDetection,
    extract_frame_array,
    list_pickle_frames,
    load_pickle_frame,
    preload_pickle_frames,
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


def test_preload_pickle_frames_loads_frames_in_order(tmp_path: Path) -> None:
    frame = np.zeros((512, 512), dtype=np.uint16)
    for name in ("0000002.pickle", "0000001.pickle"):
        with (tmp_path / name).open("wb") as file:
            pickle.dump(frame, file)

    loaded = preload_pickle_frames(tmp_path)

    assert [path.name for path, _ in loaded] == ["0000001.pickle", "0000002.pickle"]
    assert all(item.shape == (512, 512) for _, item in loaded)


def test_preload_pickle_frames_supports_progress_flag(tmp_path: Path) -> None:
    frame = np.zeros((512, 512), dtype=np.uint16)
    with (tmp_path / "0000001.pickle").open("wb") as file:
        pickle.dump(frame, file)

    loaded = preload_pickle_frames(tmp_path, progress=False)

    assert len(loaded) == 1


def test_resolve_weights_returns_bundled_weights_when_run_weights_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    weights = resolve_weights(None)

    assert weights is not None
    assert weights.exists()


def test_split_frame_paths_is_reproducible_and_sorts_each_split() -> None:
    frames = [Path(f"{index:07d}.pickle") for index in range(10, 0, -1)]

    train_frames, val_frames = split_frame_paths(frames, val_ratio=0.2, seed=7)
    train_frames_again, val_frames_again = split_frame_paths(frames, val_ratio=0.2, seed=7)

    assert train_frames == train_frames_again
    assert val_frames == val_frames_again
    assert len(train_frames) == 8
    assert len(val_frames) == 2
    assert train_frames == sorted(train_frames)
    assert val_frames == sorted(val_frames)


def test_split_frame_paths_rejects_invalid_val_ratio() -> None:
    frames = [Path("0000001.pickle")]

    with pytest.raises(ValueError):
        split_frame_paths(frames, val_ratio=1.0, seed=7)


def test_parse_args_has_no_default_detection_count_cap(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["example_infrared_video.py"])

    args = parse_args()

    assert args.conf == pytest.approx(0.25)
    assert args.max_detections == 0


def test_bbox_iou_returns_overlap_ratio() -> None:
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_merge_marker_detections_adds_only_non_duplicate_rescue_boxes() -> None:
    yolo_detections = [
        MarkerDetection((10, 10, 18, 18), confidence=0.80),
        MarkerDetection((40, 40, 48, 48), confidence=0.70),
    ]
    rescue_detections = [
        MarkerDetection((11, 11, 19, 19), confidence=0.95),
        MarkerDetection((70, 70, 78, 78), confidence=0.90),
    ]

    merged = merge_marker_detections(
        yolo_detections,
        rescue_detections,
        iou_threshold=0.10,
        center_distance=8.0,
    )

    boxes = [detection.bbox_xyxy for detection in merged]
    assert len(merged) == 3
    assert (70, 70, 78, 78) in boxes
    assert (11, 11, 19, 19) not in boxes


def test_fuse_pseudo_label_detections_keeps_confirmed_low_confidence_yolo_only() -> None:
    yolo_detections = [
        MarkerDetection((10, 10, 18, 18), confidence=0.80),
        MarkerDetection((40, 40, 48, 48), confidence=0.08),
        MarkerDetection((90, 90, 98, 98), confidence=0.07),
    ]
    circle_detections = [
        MarkerDetection((41, 41, 49, 49), confidence=0.85),
        MarkerDetection((120, 120, 128, 128), confidence=0.90),
    ]

    fused = fuse_pseudo_label_detections(
        yolo_detections,
        circle_detections,
        yolo_accept_confidence=0.25,
        circle_confidence=0.60,
        iou_threshold=0.10,
        center_distance=8.0,
    )

    boxes = [detection.bbox_xyxy for detection in fused]
    assert (10, 10, 18, 18) in boxes
    assert (40, 40, 48, 48) in boxes
    assert (90, 90, 98, 98) not in boxes
    assert (120, 120, 128, 128) in boxes


def test_preview_marker_detector_finds_bright_circle() -> None:
    frame = np.zeros((512, 512), dtype=np.uint16)
    yy, xx = np.ogrid[:512, :512]
    mask = (xx - 250) ** 2 + (yy - 200) ** 2 <= 8**2
    frame[mask] = 60_000

    detections = BrightCircleDetector().detect(frame)

    assert len(detections) == 1
    x_min, y_min, x_max, y_max = detections[0].bbox_xyxy
    assert 240 <= x_min <= 252
    assert 192 <= y_min <= 204
    assert 248 <= x_max <= 260
    assert 200 <= y_max <= 212
    assert detections[0].confidence > 0.5
