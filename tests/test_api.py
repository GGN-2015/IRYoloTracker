from __future__ import annotations

import numpy as np

from ir_yolo_tracker import (
    IRMarkerTracker,
    detect_marker_batch,
    detect_marker_dicts,
    detect_markers,
    iter_marker_detections,
    iter_pickle_detections,
)


class FakeBoxes:
    xyxy = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    conf = np.array([0.8], dtype=np.float32)
    cls = np.array([0.0], dtype=np.float32)


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def predict(self, **kwargs: object) -> list[FakeResult]:
        return [FakeResult()]


def test_detect_markers_uses_injected_tracker() -> None:
    tracker = IRMarkerTracker(model=FakeModel())
    frame = np.zeros((512, 512), dtype=np.uint16)

    detections = detect_markers(frame, tracker=tracker)

    assert len(detections) == 1
    assert detections[0].bbox_xyxy == (10.0, 20.0, 30.0, 40.0)


def test_detect_marker_dicts_returns_json_ready_dicts() -> None:
    tracker = IRMarkerTracker(model=FakeModel())
    frame = np.zeros((512, 512), dtype=np.uint16)

    detections = detect_marker_dicts(frame, tracker=tracker)

    assert detections == [
        {
            "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
            "confidence": 0.800000011920929,
            "class_id": 0,
            "class_name": "ir_marker_ball",
            "center_xy": [20.0, 30.0],
        }
    ]


def test_iter_marker_detections_reuses_tracker() -> None:
    tracker = IRMarkerTracker(model=FakeModel())
    frames = [np.zeros((512, 512), dtype=np.uint16) for _ in range(2)]

    results = list(iter_marker_detections(frames, tracker=tracker))

    assert [len(result) for result in results] == [1, 1]


def test_detect_marker_batch_returns_list() -> None:
    tracker = IRMarkerTracker(model=FakeModel())
    frames = [np.zeros((512, 512), dtype=np.uint16) for _ in range(2)]

    results = detect_marker_batch(frames, tracker=tracker)

    assert len(results) == 2


def test_iter_pickle_detections(tmp_path) -> None:
    import pickle

    frame = np.zeros((512, 512), dtype=np.uint16)
    path = tmp_path / "0000001.pickle"
    with path.open("wb") as file:
        pickle.dump(frame, file)

    tracker = IRMarkerTracker(model=FakeModel())

    results = list(iter_pickle_detections(tmp_path, tracker=tracker))

    assert results[0][0] == path
    assert len(results[0][1]) == 1
