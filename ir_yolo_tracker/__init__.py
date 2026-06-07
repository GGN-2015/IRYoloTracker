"""Infrared marker ball detection with YOLO."""

from .api import (
    create_tracker,
    detect_marker_batch,
    detect_marker_dicts,
    detect_markers,
    iter_marker_detections,
    iter_pickle_detections,
)
from .detections import MarkerDetection
from .heuristics import BrightCircleDetector
from .io import (
    extract_frame_array,
    iter_pickle_frames,
    list_pickle_frames,
    load_pickle_frame,
    preload_pickle_frames,
)
from .model_assets import DEFAULT_MODEL_NAME, get_default_model_path
from .preprocessing import PreprocessConfig, prepare_yolo_image, validate_ir_frame
from .tracker import IRMarkerTracker
from .visualization import draw_detections, resize_for_display
from .yolo_format import format_yolo_label_line, write_yolo_label_file, xyxy_to_yolo

__all__ = [
    "IRMarkerTracker",
    "BrightCircleDetector",
    "DEFAULT_MODEL_NAME",
    "MarkerDetection",
    "PreprocessConfig",
    "create_tracker",
    "detect_marker_batch",
    "detect_marker_dicts",
    "detect_markers",
    "draw_detections",
    "extract_frame_array",
    "format_yolo_label_line",
    "get_default_model_path",
    "iter_marker_detections",
    "iter_pickle_detections",
    "iter_pickle_frames",
    "list_pickle_frames",
    "load_pickle_frame",
    "prepare_yolo_image",
    "preload_pickle_frames",
    "resize_for_display",
    "validate_ir_frame",
    "write_yolo_label_file",
    "xyxy_to_yolo",
]
