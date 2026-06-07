"""YOLO-based detector for bright infrared marker balls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .detections import MarkerDetection
from .model_assets import get_default_model_path
from .preprocessing import ModelInputChannels, PreprocessConfig, prepare_yolo_image


class IRMarkerTracker:
    """Detect infrared marker balls in single-channel uint16 frames.

    The tracker only returns detections for ``marker_class_id``. Train the YOLO
    model with one class named ``ir_marker_ball`` and use class id 0.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        model: Any | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 512,
        marker_class_id: int = 0,
        marker_class_name: str = "ir_marker_ball",
        model_input_channels: ModelInputChannels = 3,
        preprocess_config: PreprocessConfig | None = None,
        device: str | int | None = "cuda",
    ) -> None:
        if model is None and weights is None:
            weights = get_default_model_path()
        if model_input_channels not in (1, 3):
            raise ValueError("model_input_channels must be 1 or 3.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")

        self.device = _resolve_device(device)
        self.weights = Path(weights) if weights is not None else None
        self.model = model if model is not None else self._load_yolo(weights)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.marker_class_id = marker_class_id
        self.marker_class_name = marker_class_name
        self.model_input_channels = model_input_channels
        self.preprocess_config = preprocess_config or PreprocessConfig()

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        """Return marker-ball bounding boxes and confidences for one IR frame."""

        yolo_image = prepare_yolo_image(
            frame,
            config=self.preprocess_config,
            model_input_channels=self.model_input_channels,
        )

        predict_kwargs: dict[str, Any] = {
            "source": yolo_image,
            "imgsz": self.image_size,
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "classes": [self.marker_class_id],
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        return self._parse_result(results[0], frame.shape)

    def detect_dicts(self, frame: np.ndarray) -> list[dict[str, object]]:
        """Return detections as JSON-serializable dictionaries."""

        return [detection.to_dict() for detection in self.detect(frame)]

    def __call__(self, frame: np.ndarray) -> list[MarkerDetection]:
        return self.detect(frame)

    @staticmethod
    def _load_yolo(weights: str | Path | None) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install it with "
                "`python -m pip install ultralytics`."
            ) from exc

        return YOLO(str(weights))

    def _parse_result(
        self,
        result: Any,
        frame_shape: tuple[int, int],
    ) -> list[MarkerDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(getattr(boxes, "xyxy", np.empty((0, 4), dtype=np.float32)))
        confidences = _to_numpy(getattr(boxes, "conf", np.empty((0,), dtype=np.float32)))
        class_ids = _to_numpy(getattr(boxes, "cls", np.zeros(len(xyxy), dtype=np.float32)))

        detections: list[MarkerDetection] = []
        for bbox, confidence, class_id in zip(xyxy, confidences, class_ids, strict=False):
            if float(confidence) < self.confidence_threshold:
                continue
            if int(class_id) != self.marker_class_id:
                continue

            clipped_bbox = _clip_xyxy(bbox, frame_shape)
            if clipped_bbox is None:
                continue

            detections.append(
                MarkerDetection(
                    bbox_xyxy=clipped_bbox,
                    confidence=float(confidence),
                    class_id=self.marker_class_id,
                    class_name=self.marker_class_name,
                )
            )

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _resolve_device(device: str | int | None) -> str | int | None:
    if device is None:
        return None
    if isinstance(device, int):
        _require_cuda()
        return device

    normalized = device.lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda") or normalized.isdigit():
        _require_cuda()
        return device
    return device


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "GPU inference requires PyTorch with CUDA support. "
            "Install a CUDA-enabled torch build, or pass device='cpu' explicitly."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU inference was requested, but CUDA is not available in this Python environment. "
            "Install a CUDA-enabled PyTorch build in this venv, or pass device='cpu' explicitly "
            "if you accept slower CPU inference."
        )


def _clip_xyxy(
    bbox: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    height, width = frame_shape
    x_min, y_min, x_max, y_max = [float(v) for v in bbox[:4]]

    x_min = min(max(x_min, 0.0), float(width))
    x_max = min(max(x_max, 0.0), float(width))
    y_min = min(max(y_min, 0.0), float(height))
    y_max = min(max(y_max, 0.0), float(height))

    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)
