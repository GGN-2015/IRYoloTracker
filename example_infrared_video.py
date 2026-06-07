r"""Play infrared pickle frames with real-time marker-ball detections.

Run from the project root:

    .\venv\Scripts\python.exe example_infrared_video.py

If trained YOLO weights are available, pass them explicitly:

    .\venv\Scripts\python.exe example_infrared_video.py --weights path\to\best.pt

Without YOLO weights, this example falls back to a simple bright-circle preview
detector so the infrared video player can still run.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from ir_yolo_tracker import (
    BrightCircleDetector,
    IRMarkerTracker,
    MarkerDetection,
    draw_detections,
    get_default_model_path,
    list_pickle_frames,
    load_pickle_frame,
    preload_pickle_frames,
)
from ir_yolo_tracker.preprocessing import normalize_uint16_to_uint8
from ir_yolo_tracker.yolo_format import write_yolo_label_file

WINDOW_NAME = "IRYoloTracker - Infrared Marker Balls"
DEFAULT_WEIGHTS = Path("runs/detect/ir_marker_ball/weights/best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("infrared_data"),
        help="Directory containing per-frame .pickle infrared images.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Trained single-class YOLO .pt weights. If omitted, the script searches runs/.",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="Playback frame rate.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cpu', 'cuda', or 0. Defaults to Ultralytics auto selection.",
    )
    parser.add_argument(
        "--model-input-channels",
        type=int,
        choices=(1, 3),
        default=3,
        help="Use 3 for standard YOLO weights or 1 for a true one-channel model.",
    )
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale factor.")
    parser.add_argument("--loop", action="store_true", help="Loop playback until Esc/Q is pressed.")
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Load each pickle frame during playback instead of preloading all frames at startup.",
    )
    parser.add_argument(
        "--require-yolo",
        action="store_true",
        help="Exit with an error instead of using the bright-circle preview detector.",
    )
    parser.add_argument(
        "--rescue-bright-circles",
        dest="rescue_bright_circles",
        action="store_true",
        default=True,
        help="Add conservative bright-circle candidates that YOLO missed.",
    )
    parser.add_argument(
        "--no-rescue-bright-circles",
        dest="rescue_bright_circles",
        action="store_false",
        help="Use raw YOLO detections only, without bright-circle rescue candidates.",
    )
    parser.add_argument(
        "--rescue-threshold-percentile",
        type=float,
        default=99.9,
        help="Percentile threshold for the conservative bright-circle rescue detector.",
    )
    parser.add_argument(
        "--rescue-confidence",
        type=float,
        default=0.60,
        help="Minimum bright-circle confidence for rescue candidates.",
    )
    parser.add_argument(
        "--rescue-merge-iou",
        type=float,
        default=0.10,
        help="IoU used to merge duplicate YOLO and rescue boxes.",
    )
    parser.add_argument(
        "--rescue-center-distance",
        type=float,
        default=8.0,
        help="Pixel center distance used to merge duplicate YOLO and rescue boxes.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=0,
        help="Keep only the top N merged detections. Use 0 to keep all detections.",
    )
    pseudo_group = parser.add_argument_group("YOLO pseudo dataset rebuild")
    pseudo_group.add_argument(
        "--rebuild-yolo-pseudo-dataset",
        action="store_true",
        help=(
            "Use the current YOLO inference model at a low confidence threshold "
            "to re-split infrared_data into a pseudo-labeled YOLO train/val dataset."
        ),
    )
    pseudo_group.add_argument(
        "--pseudo-output",
        type=Path,
        default=Path("datasets/yolo_low_conf_ir_marker_ball"),
        help="Output directory for the low-confidence YOLO pseudo dataset.",
    )
    pseudo_group.add_argument(
        "--pseudo-dataset-yaml",
        type=Path,
        default=Path("configs/yolo_low_conf_ir_marker_dataset.yaml"),
        help="Dataset YAML written for the low-confidence YOLO pseudo dataset.",
    )
    pseudo_group.add_argument(
        "--pseudo-conf",
        type=float,
        default=0.05,
        help="Low YOLO confidence threshold used only when rebuilding pseudo labels.",
    )
    pseudo_group.add_argument(
        "--pseudo-yolo-accept-conf",
        type=float,
        default=0.08,
        help="YOLO pseudo boxes at or above this confidence are kept without circle confirmation.",
    )
    pseudo_group.add_argument(
        "--pseudo-circle-confidence",
        type=float,
        default=0.60,
        help="Minimum bright-circle confidence used when rebuilding pseudo labels.",
    )
    pseudo_group.add_argument(
        "--pseudo-val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio used when rebuilding pseudo labels.",
    )
    pseudo_group.add_argument(
        "--pseudo-seed",
        type=int,
        default=42,
        help="Random seed for the pseudo dataset train/val split.",
    )
    pseudo_group.add_argument(
        "--keep-pseudo-output",
        action="store_true",
        help="Keep existing pseudo dataset files instead of cleaning train/val folders first.",
    )
    return parser.parse_args()


def resolve_weights(explicit_weights: Path | None) -> Path | None:
    if explicit_weights is not None:
        if not explicit_weights.exists():
            raise FileNotFoundError(f"YOLO weights do not exist: {explicit_weights}")
        return explicit_weights

    env_weights = os.environ.get("IR_YOLO_WEIGHTS")
    if env_weights:
        path = Path(env_weights)
        if path.exists():
            return path
        raise FileNotFoundError(f"IR_YOLO_WEIGHTS points to a missing file: {path}")

    if DEFAULT_WEIGHTS.exists():
        return DEFAULT_WEIGHTS

    bundled_weights = get_default_model_path()
    if bundled_weights.exists():
        return bundled_weights

    candidates = sorted(
        Path("runs").glob("**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    return None


class YoloDetector:
    """Small adapter that gives the YOLO tracker a display name."""

    mode_name = "YOLO"

    def __init__(self, tracker: IRMarkerTracker) -> None:
        self.tracker = tracker

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        return self.tracker.detect(frame)


class YoloWithCircleRescueDetector:
    """YOLO detector with a conservative bright-circle rescue pass."""

    mode_name = "YOLO+circle"

    def __init__(
        self,
        tracker: IRMarkerTracker,
        *,
        rescue_detector: BrightCircleDetector,
        rescue_confidence: float,
        merge_iou: float,
        center_distance: float,
        max_detections: int | None,
    ) -> None:
        self.tracker = tracker
        self.rescue_detector = rescue_detector
        self.rescue_confidence = rescue_confidence
        self.merge_iou = merge_iou
        self.center_distance = center_distance
        self.max_detections = max_detections

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        yolo_detections = self.tracker.detect(frame)
        rescue_detections = [
            detection
            for detection in self.rescue_detector.detect(frame)
            if detection.confidence >= self.rescue_confidence
        ]
        return merge_marker_detections(
            yolo_detections,
            rescue_detections,
            iou_threshold=self.merge_iou,
            center_distance=self.center_distance,
            max_detections=self.max_detections,
        )


def create_detector(
    args: argparse.Namespace,
) -> tuple[YoloDetector | YoloWithCircleRescueDetector | BrightCircleDetector, Path | None]:
    weights = resolve_weights(args.weights)
    if weights is None:
        if args.require_yolo:
            raise FileNotFoundError(
                "No trained YOLO weights found. Train the marker-ball model first, "
                "or run this example with --weights path\\to\\best.pt."
            )
        print("No YOLO weights found. Using the bright-circle preview detector.")
        return BrightCircleDetector(), None

    tracker = IRMarkerTracker(
        weights,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        model_input_channels=args.model_input_channels,
        device=args.device,
    )
    return wrap_yolo_tracker_for_display(tracker, args), weights


def wrap_yolo_tracker_for_display(
    tracker: IRMarkerTracker,
    args: argparse.Namespace,
) -> YoloDetector | YoloWithCircleRescueDetector:
    if not args.rescue_bright_circles:
        return YoloDetector(tracker)

    return YoloWithCircleRescueDetector(
        tracker,
        rescue_detector=BrightCircleDetector(
            threshold_percentile=args.rescue_threshold_percentile,
        ),
        rescue_confidence=args.rescue_confidence,
        merge_iou=args.rescue_merge_iou,
        center_distance=args.rescue_center_distance,
        max_detections=normalize_max_detections(args.max_detections),
    )


def normalize_max_detections(value: int) -> int | None:
    if value < 0:
        raise ValueError("--max-detections must be 0 or a positive integer.")
    return value or None


def create_pseudo_label_detector(
    args: argparse.Namespace,
) -> tuple[YoloDetector, BrightCircleDetector, Path]:
    weights = resolve_weights(args.weights)
    if weights is None:
        raise FileNotFoundError(
            "No YOLO weights found. The pseudo dataset rebuild must use an existing "
            "YOLO model, so pass --weights path\\to\\best.pt or keep the bundled model installed."
        )

    if not 0.0 <= args.pseudo_conf <= 1.0:
        raise ValueError("--pseudo-conf must be between 0 and 1.")
    if not 0.0 <= args.pseudo_yolo_accept_conf <= 1.0:
        raise ValueError("--pseudo-yolo-accept-conf must be between 0 and 1.")
    if args.pseudo_yolo_accept_conf < args.pseudo_conf:
        raise ValueError("--pseudo-yolo-accept-conf must be greater than or equal to --pseudo-conf.")
    if not 0.0 <= args.pseudo_circle_confidence <= 1.0:
        raise ValueError("--pseudo-circle-confidence must be between 0 and 1.")

    tracker = IRMarkerTracker(
        weights,
        confidence_threshold=args.pseudo_conf,
        iou_threshold=args.iou,
        model_input_channels=args.model_input_channels,
        device=args.device,
    )
    circle_detector = BrightCircleDetector(
        threshold_percentile=args.rescue_threshold_percentile,
    )
    return YoloDetector(tracker), circle_detector, weights


def merge_marker_detections(
    primary: list[MarkerDetection],
    rescue: list[MarkerDetection],
    *,
    iou_threshold: float,
    center_distance: float,
    max_detections: int | None = None,
) -> list[MarkerDetection]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1.")
    if center_distance < 0:
        raise ValueError("center_distance must be non-negative.")

    merged = list(primary)
    for candidate in rescue:
        if any(
            detections_match(
                candidate,
                existing,
                iou_threshold=iou_threshold,
                center_distance=center_distance,
            )
            for existing in merged
        ):
            continue
        merged.append(candidate)

    merged.sort(key=lambda detection: detection.confidence, reverse=True)
    if max_detections is not None:
        merged = merged[:max_detections]
    return merged


def fuse_pseudo_label_detections(
    yolo_detections: list[MarkerDetection],
    circle_detections: list[MarkerDetection],
    *,
    yolo_accept_confidence: float,
    circle_confidence: float,
    iou_threshold: float,
    center_distance: float,
) -> list[MarkerDetection]:
    """Fuse YOLO and bright-circle evidence for pseudo labels.

    High-confidence YOLO boxes are kept directly. Low-confidence YOLO boxes are
    kept only when a bright circular blob confirms the same marker location.
    High-confidence circle detections can still rescue a missed marker.
    """

    confirmed_circles = [
        detection
        for detection in circle_detections
        if detection.confidence >= circle_confidence
    ]
    accepted: list[MarkerDetection] = []

    for detection in yolo_detections:
        if detection.confidence >= yolo_accept_confidence or any(
            detections_match(
                detection,
                circle,
                iou_threshold=iou_threshold,
                center_distance=center_distance,
            )
            for circle in confirmed_circles
        ):
            accepted.append(detection)

    return merge_marker_detections(
        accepted,
        confirmed_circles,
        iou_threshold=iou_threshold,
        center_distance=center_distance,
    )


def detections_match(
    first: MarkerDetection,
    second: MarkerDetection,
    *,
    iou_threshold: float,
    center_distance: float,
) -> bool:
    if bbox_iou(first.bbox_xyxy, second.bbox_xyxy) >= iou_threshold:
        return True

    first_x, first_y = first.center_xy
    second_x, second_y = second.center_xy
    distance = ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5
    return distance <= center_distance


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second

    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def split_frame_paths(
    frame_paths: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--pseudo-val-ratio must satisfy 0 <= ratio < 1.")

    shuffled = list(frame_paths)
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    val_frames = sorted(shuffled[:val_count])
    train_frames = sorted(shuffled[val_count:])
    return train_frames, val_frames


def rebuild_yolo_pseudo_dataset(args: argparse.Namespace) -> None:
    frame_paths = list_pickle_frames(args.data_dir)
    train_frames, val_frames = split_frame_paths(
        frame_paths,
        val_ratio=args.pseudo_val_ratio,
        seed=args.pseudo_seed,
    )
    detector, circle_detector, weights = create_pseudo_label_detector(args)

    prepare_pseudo_output_dirs(args.pseudo_output, clean=not args.keep_pseudo_output)

    print(f"Using YOLO weights: {weights}")
    print(f"Pseudo confidence threshold: {args.pseudo_conf:.3f}")
    print(
        f"Split: {len(train_frames)} train frames, {len(val_frames)} val frames "
        f"from {args.data_dir}"
    )

    if frame_paths:
        warmup_frame = load_pickle_frame(frame_paths[0])
        detector.detect(warmup_frame)
        circle_detector.detect(warmup_frame)

    train_labeled, train_boxes = write_pseudo_dataset_split(
        train_frames,
        split="train",
        output_root=args.pseudo_output,
        detector=detector,
        circle_detector=circle_detector,
        args=args,
    )
    val_labeled, val_boxes = write_pseudo_dataset_split(
        val_frames,
        split="val",
        output_root=args.pseudo_output,
        detector=detector,
        circle_detector=circle_detector,
        args=args,
    )
    write_pseudo_dataset_yaml(args.pseudo_dataset_yaml, args.pseudo_output)

    print(f"Dataset: {args.pseudo_output}")
    print(f"YAML: {args.pseudo_dataset_yaml}")
    print(f"Train: {len(train_frames)} frames, {train_labeled} labeled, {train_boxes} boxes")
    print(f"Val: {len(val_frames)} frames, {val_labeled} labeled, {val_boxes} boxes")


def prepare_pseudo_output_dirs(output_root: Path, *, clean: bool) -> None:
    for split in ("train", "val"):
        for folder in ("images", "labels"):
            directory = output_root / folder / split
            if clean and directory.exists():
                remove_generated_directory(directory, output_root)
            directory.mkdir(parents=True, exist_ok=True)


def remove_generated_directory(directory: Path, output_root: Path) -> None:
    resolved_directory = directory.resolve()
    resolved_root = output_root.resolve()
    if resolved_directory == resolved_root or resolved_root not in resolved_directory.parents:
        raise ValueError(f"Refusing to remove directory outside pseudo output root: {directory}")
    shutil.rmtree(resolved_directory)


def write_pseudo_dataset_split(
    frame_paths: list[Path],
    *,
    split: str,
    output_root: Path,
    detector: YoloDetector,
    circle_detector: BrightCircleDetector,
    args: argparse.Namespace,
) -> tuple[int, int]:
    images_dir = output_root / "images" / split
    labels_dir = output_root / "labels" / split

    frames_with_labels = 0
    total_boxes = 0
    for frame_path in progress_frames(frame_paths, description=f"YOLO pseudo-label {split}"):
        frame = load_pickle_frame(frame_path)
        detections = fuse_pseudo_label_detections(
            detector.detect(frame),
            circle_detector.detect(frame),
            yolo_accept_confidence=args.pseudo_yolo_accept_conf,
            circle_confidence=args.pseudo_circle_confidence,
            iou_threshold=args.rescue_merge_iou,
            center_distance=args.rescue_center_distance,
        )

        if detections:
            frames_with_labels += 1
        total_boxes += len(detections)

        gray = normalize_uint16_to_uint8(frame)
        image_path = images_dir / f"{frame_path.stem}.png"
        label_path = labels_dir / f"{frame_path.stem}.txt"

        if not cv2.imwrite(str(image_path), gray):
            raise OSError(f"failed to write image: {image_path}")

        write_yolo_label_file(
            label_path,
            [detection.bbox_xyxy for detection in detections],
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            class_id=0,
        )

    return frames_with_labels, total_boxes


def progress_frames(frame_paths: list[Path], *, description: str):
    try:
        from tqdm import tqdm
    except ImportError:
        return frame_paths
    return tqdm(frame_paths, desc=description, unit="frame")


def write_pseudo_dataset_yaml(dataset_yaml: Path, output_root: Path) -> None:
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Low-confidence YOLO pseudo dataset generated by example_infrared_video.py.\n"
        "# Labels come from the current inference model, not human annotation.\n\n"
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test:\n\n"
        "names:\n"
        "  0: ir_marker_ball\n"
    )
    dataset_yaml.write_text(content, encoding="utf-8")


def play_frames(
    frames: list[tuple[Path, np.ndarray | None]],
    detector: YoloDetector | YoloWithCircleRescueDetector | BrightCircleDetector,
    fps: float,
    display_scale: float,
    loop: bool,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive.")

    frame_interval = 1.0 / fps
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_frame_time = time.perf_counter()
    try:
        while True:
            for index, (frame_path, frame) in enumerate(frames):
                frame_start = time.perf_counter()

                if frame is None:
                    frame = load_pickle_frame(frame_path)

                detections = detector.detect(frame)

                now = time.perf_counter()
                actual_fps = 1.0 / max(now - last_frame_time, 1e-9)
                last_frame_time = now

                status = (
                    f"{detector.mode_name}  {index + 1}/{len(frames)}  {frame_path.name}  "
                    f"detections: {len(detections)}  fps: {actual_fps:4.1f}"
                )
                display = draw_detections(
                    frame,
                    detections,
                    status=status,
                    scale=display_scale,
                )
                cv2.imshow(WINDOW_NAME, display)

                elapsed = time.perf_counter() - frame_start
                delay_ms = max(1, int(round((frame_interval - elapsed) * 1000.0)))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    return
                if key == ord(" ") and not wait_until_resume():
                    return

            if not loop:
                return
    finally:
        cv2.destroyAllWindows()


def wait_until_resume() -> bool:
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            return False
        if key == ord(" "):
            return True


def main() -> None:
    args = parse_args()
    if args.rebuild_yolo_pseudo_dataset:
        rebuild_yolo_pseudo_dataset(args)
        return

    detector, weights = create_detector(args)

    if weights is not None:
        print(f"Using YOLO weights: {weights}")

    if args.lazy_load:
        frame_paths = list_pickle_frames(args.data_dir)
        frames: list[tuple[Path, np.ndarray | None]] = [(path, None) for path in frame_paths]
        print(f"Lazy-load mode: {len(frame_paths)} pickle frames will be read during playback.")
    else:
        preload_start = time.perf_counter()
        frames = preload_pickle_frames(args.data_dir, progress=True)
        elapsed = time.perf_counter() - preload_start
        total_bytes = sum(frame.nbytes for _, frame in frames)
        print(
            f"Preloaded {len(frames)} pickle frames from {args.data_dir} "
            f"in {elapsed:.2f}s ({total_bytes / (1024 * 1024):.1f} MiB)."
        )

    if frames:
        warmup_start = time.perf_counter()
        first_frame = frames[0][1] if frames[0][1] is not None else load_pickle_frame(frames[0][0])
        detector.detect(first_frame)
        print(f"Detector warm-up finished in {time.perf_counter() - warmup_start:.2f}s.")

    print("Press Space to pause/resume, Q or Esc to quit.")

    play_frames(
        frames=frames,
        detector=detector,
        fps=args.fps,
        display_scale=args.scale,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
