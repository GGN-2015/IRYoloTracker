from __future__ import annotations

from pathlib import Path

import pytest

from ir_yolo_tracker.yolo_format import (
    format_yolo_label_line,
    write_yolo_label_file,
    xyxy_to_yolo,
)


def test_xyxy_to_yolo_normalizes_box() -> None:
    assert xyxy_to_yolo((128, 128, 256, 384), 512, 512) == pytest.approx(
        (0.375, 0.5, 0.25, 0.5)
    )


def test_xyxy_to_yolo_clips_to_image_bounds() -> None:
    assert xyxy_to_yolo((-10, -20, 20, 30), 512, 512) == pytest.approx(
        (0.01953125, 0.029296875, 0.0390625, 0.05859375)
    )


def test_format_yolo_label_line_uses_class_zero() -> None:
    assert format_yolo_label_line((0, 0, 512, 512), 512, 512) == (
        "0 0.50000000 0.50000000 1.00000000 1.00000000"
    )


def test_write_yolo_label_file(tmp_path: Path) -> None:
    label_path = tmp_path / "frame.txt"

    write_yolo_label_file(label_path, [(0, 0, 128, 128)], 512, 512)

    assert label_path.read_text(encoding="utf-8") == (
        "0 0.12500000 0.12500000 0.25000000 0.25000000\n"
    )
