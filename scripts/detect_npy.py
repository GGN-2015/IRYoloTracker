"""Run infrared marker-ball detection on a uint16 .npy frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ir_yolo_tracker import IRMarkerTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path, help="Path to a 512x512 uint16 .npy file.")
    parser.add_argument("--weights", required=True, type=Path, help="Trained YOLO .pt weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--model-input-channels",
        type=int,
        choices=(1, 3),
        default=3,
        help="Use 3 for standard YOLO weights or 1 for a one-channel YOLO model.",
    )
    parser.add_argument("--device", default=None, help="CUDA device id, 'cpu', or None.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = np.load(args.frame)

    tracker = IRMarkerTracker(
        args.weights,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        model_input_channels=args.model_input_channels,
        device=args.device,
    )
    print(json.dumps(tracker.detect_dicts(frame), indent=2))


if __name__ == "__main__":
    main()
