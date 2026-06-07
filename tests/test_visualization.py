from __future__ import annotations

import numpy as np

from ir_yolo_tracker import MarkerDetection, draw_detections


def test_draw_detections_returns_bgr_image() -> None:
    frame = np.zeros((240, 320), dtype=np.uint16)
    detections = [
        MarkerDetection(
            bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
            confidence=0.8,
        )
    ]

    image = draw_detections(frame, detections, status="test")

    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8
    assert image.max() > 0


def test_draw_detections_scales_output() -> None:
    frame = np.zeros((240, 320), dtype=np.uint16)

    image = draw_detections(frame, [], scale=0.5)

    assert image.shape == (120, 160, 3)
