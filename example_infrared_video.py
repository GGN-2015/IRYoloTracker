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
import pickle
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ir_yolo_tracker import IRMarkerTracker, MarkerDetection, get_default_model_path
from ir_yolo_tracker.preprocessing import normalize_uint16_to_uint8, validate_ir_frame

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
        default="cuda",
        help="Inference device. Defaults to 'cuda'. Use 'cpu' only if you accept slower playback.",
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
        "--require-yolo",
        action="store_true",
        help="Exit with an error instead of using the bright-circle preview detector.",
    )
    return parser.parse_args()


def list_pickle_frames(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    frames = sorted(data_dir.glob("*.pickle"))
    if not frames:
        raise FileNotFoundError(f"No .pickle frames found in: {data_dir}")
    return frames


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


class PreviewMarkerDetector:
    """Lightweight bright-circle detector used only by this runnable example."""

    mode_name = "preview"

    def __init__(
        self,
        *,
        min_area: int = 8,
        max_area: int = 2_500,
        min_circularity: float = 0.45,
        threshold_percentile: float = 99.7,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.threshold_percentile = threshold_percentile

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        validate_ir_frame(frame)
        gray = normalize_uint16_to_uint8(frame)
        threshold = max(1, int(np.percentile(gray, self.threshold_percentile)))
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[MarkerDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = float(cv2.arcLength(contour, closed=True))
            if perimeter <= 0.0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            aspect_ratio = width / max(height, 1)
            if not 0.55 <= aspect_ratio <= 1.8:
                continue

            confidence = min(1.0, max(0.0, circularity) * (float(gray[y : y + height, x : x + width].max()) / 255.0))
            detections.append(
                MarkerDetection(
                    bbox_xyxy=(float(x), float(y), float(x + width), float(y + height)),
                    confidence=confidence,
                    class_id=0,
                    class_name="ir_marker_ball",
                )
            )

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections


class YoloDetector:
    """Small adapter that gives the YOLO tracker a display name."""

    mode_name = "YOLO"

    def __init__(self, tracker: IRMarkerTracker) -> None:
        self.tracker = tracker

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        return self.tracker.detect(frame)


def create_detector(args: argparse.Namespace) -> tuple[YoloDetector | PreviewMarkerDetector, Path | None]:
    weights = resolve_weights(args.weights)
    if weights is None:
        if args.require_yolo:
            raise FileNotFoundError(
                "No trained YOLO weights found. Train the marker-ball model first, "
                "or run this example with --weights path\\to\\best.pt."
            )
        print("No YOLO weights found. Using the bright-circle preview detector.")
        return PreviewMarkerDetector(), None

    tracker = IRMarkerTracker(
        weights,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        model_input_channels=args.model_input_channels,
        device=args.device,
    )
    return YoloDetector(tracker), weights


def load_pickle_frame(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        payload = pickle.load(file)

    frame = extract_frame_array(payload)
    validate_ir_frame(frame)
    return frame


def extract_frame_array(payload: Any) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        return payload

    if isinstance(payload, dict):
        preferred_keys = ("frame", "image", "infrared", "ir", "data", "array")
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, np.ndarray):
                return value
        for value in payload.values():
            if isinstance(value, np.ndarray):
                return value

    if isinstance(payload, (list, tuple)):
        for value in payload:
            if isinstance(value, np.ndarray):
                return value

    raise TypeError(f"Cannot find a numpy.ndarray infrared frame in pickle payload {type(payload)!r}.")


def draw_detections(
    gray_u8: np.ndarray,
    detections: list[MarkerDetection],
    frame_name: str,
    frame_index: int,
    total_frames: int,
    actual_fps: float,
    detector_name: str,
) -> np.ndarray:
    canvas = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    for detection in detections:
        x_min, y_min, x_max, y_max = [int(round(value)) for value in detection.bbox_xyxy]
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        label = f"{detection.confidence:.2f}"
        text_origin = (x_min, max(16, y_min - 6))
        draw_label(canvas, label, text_origin)

    status = (
        f"{detector_name}  {frame_index + 1}/{total_frames}  {frame_name}  "
        f"detections: {len(detections)}  fps: {actual_fps:4.1f}"
    )
    draw_label(canvas, status, (8, 20), background=(0, 0, 0), foreground=(255, 255, 255))
    return canvas


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    background: tuple[int, int, int] = (0, 80, 0),
    foreground: tuple[int, int, int] = (255, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, min(x, image.shape[1] - width - 4))
    y = max(height + 4, min(y, image.shape[0] - baseline - 2))

    cv2.rectangle(
        image,
        (x - 2, y - height - 4),
        (x + width + 2, y + baseline + 2),
        background,
        thickness=-1,
    )
    cv2.putText(image, text, (x, y), font, font_scale, foreground, thickness, cv2.LINE_AA)


def resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if scale == 1.0:
        return image
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def play_frames(
    frame_paths: list[Path],
    detector: YoloDetector | PreviewMarkerDetector,
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
            for index, frame_path in enumerate(frame_paths):
                frame_start = time.perf_counter()

                frame = load_pickle_frame(frame_path)
                detections = detector.detect(frame)
                gray_u8 = normalize_uint16_to_uint8(frame)

                now = time.perf_counter()
                actual_fps = 1.0 / max(now - last_frame_time, 1e-9)
                last_frame_time = now

                display = draw_detections(
                    gray_u8,
                    detections,
                    frame_name=frame_path.name,
                    frame_index=index,
                    total_frames=len(frame_paths),
                    actual_fps=actual_fps,
                    detector_name=detector.mode_name,
                )
                display = resize_for_display(display, display_scale)
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
    detector, weights = create_detector(args)
    frame_paths = list_pickle_frames(args.data_dir)

    if weights is not None:
        print(f"Using YOLO weights: {weights}")
    print(f"Loaded {len(frame_paths)} pickle frames from {args.data_dir}")
    print("Press Space to pause/resume, Q or Esc to quit.")

    play_frames(
        frame_paths=frame_paths,
        detector=detector,
        fps=args.fps,
        display_scale=args.scale,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
