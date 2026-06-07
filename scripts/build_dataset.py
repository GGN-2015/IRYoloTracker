"""Build a YOLO dataset from uint16 .npy infrared frames and xyxy annotations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ir_yolo_tracker.preprocessing import PreprocessConfig, normalize_uint16_to_uint8
from ir_yolo_tracker.yolo_format import write_yolo_label_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="JSON or JSONL annotations with image/npy path and marker boxes.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("."),
        help="Base directory for relative frame paths in the annotation file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/ir_marker_ball"),
        help="Output YOLO dataset directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_annotation_records(path: Path) -> list[dict[str, Any]]:
    """Load records from JSON list/object or JSONL."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("frames", data.get("annotations", []))
    if not isinstance(data, list):
        raise ValueError("annotations must be a list or an object containing frames/annotations.")
    return data


def record_frame_path(record: dict[str, Any]) -> Path:
    value = record.get("image", record.get("frame", record.get("path")))
    if value is None:
        raise ValueError(f"annotation record has no image/frame/path field: {record}")
    return Path(str(value))


def record_boxes(record: dict[str, Any]) -> list[list[float]]:
    boxes = record.get("boxes", record.get("bboxes", record.get("bbox_xyxy", [])))
    if boxes is None:
        return []
    if isinstance(boxes, list) and len(boxes) == 4 and all(
        isinstance(value, int | float) for value in boxes
    ):
        return [[float(value) for value in boxes]]
    return [[float(value) for value in box] for box in boxes]


def split_records(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def write_split(
    records: list[dict[str, Any]],
    split: str,
    frames_root: Path,
    output_root: Path,
) -> None:
    images_dir = output_root / "images" / split
    labels_dir = output_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    config = PreprocessConfig()
    for index, record in enumerate(records):
        frame_path = frames_root / record_frame_path(record)
        frame = np.load(frame_path)
        gray = normalize_uint16_to_uint8(frame, config)

        stem = f"{frame_path.stem}_{index:06d}"
        image_path = images_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.txt"

        if not cv2.imwrite(str(image_path), gray):
            raise OSError(f"failed to write image: {image_path}")

        write_yolo_label_file(
            label_path,
            record_boxes(record),
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            class_id=0,
        )


def main() -> None:
    args = parse_args()
    records = load_annotation_records(args.annotations)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)

    write_split(train_records, "train", args.frames_root, args.output)
    write_split(val_records, "val", args.frames_root, args.output)

    print(f"Wrote {len(train_records)} train and {len(val_records)} val images to {args.output}")


if __name__ == "__main__":
    main()
