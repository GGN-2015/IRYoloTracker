from __future__ import annotations

import numpy as np
import pytest

from ir_yolo_tracker.preprocessing import PreprocessConfig, prepare_yolo_image, validate_ir_frame


def test_validate_ir_frame_accepts_any_2d_uint16_grayscale() -> None:
    frame = np.zeros((240, 320), dtype=np.uint16)

    validate_ir_frame(frame)


@pytest.mark.parametrize(
    ("frame", "expected_error"),
    [
        (np.zeros((512, 512, 1), dtype=np.uint16), ValueError),
        (np.zeros((512, 512, 3), dtype=np.uint16), ValueError),
        (np.zeros((0, 512), dtype=np.uint16), ValueError),
        (np.zeros((512, 512), dtype=np.uint8), TypeError),
    ],
)
def test_validate_ir_frame_rejects_non_matching_input(
    frame: np.ndarray,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        validate_ir_frame(frame)


def test_validate_ir_frame_can_enforce_explicit_shape() -> None:
    frame = np.zeros((240, 320), dtype=np.uint16)

    validate_ir_frame(frame, frame_shape=(240, 320))

    with pytest.raises(ValueError):
        validate_ir_frame(frame, frame_shape=(512, 512))


def test_prepare_yolo_image_repeats_gray_for_standard_yolo() -> None:
    frame = np.zeros((240, 320), dtype=np.uint16)
    frame[100:120, 150:170] = 50_000

    image = prepare_yolo_image(frame)

    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8
    assert np.array_equal(image[:, :, 0], image[:, :, 1])
    assert np.array_equal(image[:, :, 1], image[:, :, 2])
    assert image[110, 160, 0] == 255


def test_prepare_yolo_image_preserves_very_small_bright_marker() -> None:
    frame = np.zeros((512, 512), dtype=np.uint16)
    frame[250:255, 250:255] = 60_000

    image = prepare_yolo_image(frame)

    assert image[252, 252, 0] == 255
    assert image.max() == 255


def test_prepare_yolo_image_supports_true_one_channel_model_input() -> None:
    frame = np.zeros((240, 320), dtype=np.uint16)

    image = prepare_yolo_image(frame, model_input_channels=1)

    assert image.shape == (240, 320, 1)
    assert image.dtype == np.uint8


def test_percentile_config_rejects_invalid_range() -> None:
    with pytest.raises(ValueError):
        PreprocessConfig(lower_percentile=99.0, upper_percentile=1.0)
