from __future__ import annotations

import numpy as np
import pytest
import torch

from ir_yolo_tracker import IRMarkerTracker, get_default_model_path


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array(
            [
                [10.0, 20.0, 30.0, 40.0],
                [100.0, 110.0, 130.0, 145.0],
                [-10.0, -5.0, 15.0, 20.0],
            ],
            dtype=np.float32,
        )
        self.conf = np.array([0.8, 0.95, 0.2], dtype=np.float32)
        self.cls = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[FakeResult]:
        self.calls.append(kwargs)
        return [FakeResult()]


def test_default_model_asset_exists() -> None:
    assert get_default_model_path().exists()


def test_default_tracker_requires_cuda_inference_by_default() -> None:
    if torch.cuda.is_available():
        tracker = IRMarkerTracker()
        assert tracker.device == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            IRMarkerTracker()


def test_bundled_tracker_can_be_loaded_on_cpu_when_explicit() -> None:
    tracker = IRMarkerTracker(device="cpu")

    assert tracker.weights == get_default_model_path()


def test_tracker_accepts_injected_model_without_weights() -> None:
    tracker = IRMarkerTracker(model=FakeModel(), device="cpu")

    assert tracker.model is not None


def test_detect_filters_to_marker_class_and_sorts_by_confidence() -> None:
    model = FakeModel()
    tracker = IRMarkerTracker(model=model, confidence_threshold=0.1, device="cpu")
    frame = np.zeros((512, 512), dtype=np.uint16)

    detections = tracker.detect(frame)

    assert len(detections) == 2
    assert [d.class_id for d in detections] == [0, 0]
    assert detections[0].confidence == pytest.approx(0.8)
    assert detections[1].confidence == pytest.approx(0.2)
    assert detections[0].bbox_xyxy == (10.0, 20.0, 30.0, 40.0)
    assert detections[1].bbox_xyxy == (0.0, 0.0, 15.0, 20.0)


def test_detect_applies_confidence_threshold_after_backend_prediction() -> None:
    model = FakeModel()
    tracker = IRMarkerTracker(model=model, confidence_threshold=0.25, device="cpu")
    frame = np.zeros((512, 512), dtype=np.uint16)

    detections = tracker.detect(frame)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.8)


def test_detect_passes_single_marker_class_filter_to_yolo() -> None:
    model = FakeModel()
    tracker = IRMarkerTracker(
        model=model,
        confidence_threshold=0.4,
        iou_threshold=0.3,
        device="cpu",
    )
    frame = np.zeros((512, 512), dtype=np.uint16)

    tracker.detect(frame)

    call = model.calls[-1]
    assert call["classes"] == [0]
    assert call["conf"] == 0.4
    assert call["iou"] == 0.3
    assert call["verbose"] is False
    assert call["source"].shape == (512, 512, 3)


def test_detect_supports_one_channel_model_input() -> None:
    model = FakeModel()
    tracker = IRMarkerTracker(model=model, model_input_channels=1, device="cpu")
    frame = np.zeros((512, 512), dtype=np.uint16)

    tracker.detect(frame)

    assert model.calls[-1]["source"].shape == (512, 512, 1)


def test_detect_dicts_returns_json_ready_output() -> None:
    model = FakeModel()
    tracker = IRMarkerTracker(model=model, device="cpu")
    frame = np.zeros((512, 512), dtype=np.uint16)

    detections = tracker.detect_dicts(frame)

    assert detections[0]["class_name"] == "ir_marker_ball"
    assert detections[0]["bbox_xyxy"] == [10.0, 20.0, 30.0, 40.0]
    assert detections[0]["center_xy"] == [20.0, 30.0]
