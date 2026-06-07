"""Create a pseudo-labeled YOLO dataset from infrared pickle frames."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ir_yolo_tracker import BrightCircleDetector, load_pickle_frame
from ir_yolo_tracker.preprocessing import normalize_uint16_to_uint8
from ir_yolo_tracker.yolo_format import write_yolo_label_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("infrared_data"))
    parser.add_argument("--output", type=Path, default=Path("datasets/pseudo_ir_marker_ball"))
    parser.add_argument("--dataset-yaml", type=Path, default=Path("configs/pseudo_ir_marker_dataset.yaml"))
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_frames(data_dir: Path) -> list[Path]:
    frames = sorted(data_dir.glob("*.pickle"))
    if not frames:
        raise FileNotFoundError(f"No .pickle files found in {data_dir}")
    return frames


def split_frames(frames: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1.")

    shuffled = list(frames)
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def write_dataset_split(
    frame_paths: list[Path],
    split: str,
    output_root: Path,
    detector: BrightCircleDetector,
    confidence: float,
) -> tuple[int, int]:
    images_dir = output_root / "images" / split
    labels_dir = output_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    frames_with_labels = 0
    total_boxes = 0
    for frame_path in frame_paths:
        frame = load_pickle_frame(frame_path)
        detections = [
            detection
            for detection in detector.detect(frame)
            if detection.confidence >= confidence
        ]

        if detections:
            frames_with_labels += 1
        total_boxes += len(detections)

        gray = normalize_uint16_to_uint8(frame)
        image_path = images_dir / f"{frame_path.stem}.png"
        label_path = labels_dir / f"{frame_path.stem}.txt"

        if not cv2.imwrite(str(image_path), gray):
            raise OSError(f"failed to write image: {image_path}")

        boxes = [detection.bbox_xyxy for detection in detections]
        write_yolo_label_file(
            label_path,
            boxes,
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            class_id=0,
        )

    return frames_with_labels, total_boxes


def write_dataset_yaml(dataset_yaml: Path, output_root: Path) -> None:
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Pseudo-labeled bootstrap dataset generated from infrared_data.\n"
        "# Labels come from the bright-circle preview detector, not human annotation.\n\n"
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test:\n\n"
        "names:\n"
        "  0: ir_marker_ball\n"
    )
    dataset_yaml.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    frames = list_frames(args.data_dir)
    train_frames, val_frames = split_frames(frames, args.val_ratio, args.seed)
    detector = BrightCircleDetector()

    train_labeled, train_boxes = write_dataset_split(
        train_frames,
        "train",
        args.output,
        detector,
        args.confidence,
    )
    val_labeled, val_boxes = write_dataset_split(
        val_frames,
        "val",
        args.output,
        detector,
        args.confidence,
    )
    write_dataset_yaml(args.dataset_yaml, args.output)

    print(f"Dataset: {args.output}")
    print(f"YAML: {args.dataset_yaml}")
    print(f"Train: {len(train_frames)} frames, {train_labeled} labeled, {train_boxes} boxes")
    print(f"Val: {len(val_frames)} frames, {val_labeled} labeled, {val_boxes} boxes")


if __name__ == "__main__":
    main()
