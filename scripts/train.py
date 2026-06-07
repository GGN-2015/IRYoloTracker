"""Train a single-class YOLO detector for infrared marker balls."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/ir_marker_dataset.yaml"),
        help="YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Base YOLO model or YAML architecture.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="CUDA device id, 'cpu', or None.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="ir_marker_ball")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        single_cls=True,
    )


if __name__ == "__main__":
    main()
