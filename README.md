# IRYoloTracker

IRYoloTracker detects bright circular infrared marker balls in single-channel
`512x512` `numpy.uint16` frames using a single-class YOLO model.

The public API accepts grayscale infrared intensity images only. If you use
standard YOLO weights, the normalized gray image is copied into three identical
channels internally because standard Ultralytics YOLO models expect 3-channel
input. This does not add color information. If you train a true one-channel YOLO
model, pass `model_input_channels=1`.

## Install

Use the virtual environment in this repository:

```powershell
.\venv\Scripts\python.exe -m pip install -e ".[dev]"
```

For GPU inference on Windows, install the CUDA PyTorch wheels in the same venv:

```powershell
.\venv\Scripts\python.exe -m pip install --force-reinstall torch==2.12.0+cu130 torchvision==0.27.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

## Python API

The package includes a bootstrap YOLO model trained from high-confidence
bright-circle pseudo-labels in `infrared_data`. You can use it directly:

```python
import numpy as np

from ir_yolo_tracker import IRMarkerTracker

frame: np.ndarray = np.load("frame.npy")

tracker = IRMarkerTracker(confidence_threshold=0.35, device="cpu")

detections = tracker.detect_dicts(frame)
print(detections)
```

For one-off detection, use the convenience function:

```python
import numpy as np

from ir_yolo_tracker import detect_marker_dicts

frame: np.ndarray = np.load("frame.npy")
detections = detect_marker_dicts(frame, confidence_threshold=0.35, device="cpu")
```

For repeated frames, create one tracker and reuse it:

```python
from ir_yolo_tracker import IRMarkerTracker, preload_pickle_frames

tracker = IRMarkerTracker(device="cpu")
frames = preload_pickle_frames("infrared_data", progress=True)

for path, frame in frames:
    detections = tracker.detect(frame)
    print(path.name, detections)
```

Batch-style helpers are also available:

```python
from ir_yolo_tracker import detect_marker_batch, iter_pickle_detections

all_results = detect_marker_batch(frames, device="cpu")

for path, detections in iter_pickle_detections("infrared_data", device="cpu"):
    ...
```

To draw results on a frame:

```python
import cv2

from ir_yolo_tracker import draw_detections

image = draw_detections(frame, detections, status="IR marker detections")
cv2.imshow("detections", image)
cv2.waitKey(0)
```

Pass `device="cuda"` or `device=0` to request GPU inference. If `device` is not
set, Ultralytics chooses the device.

The public API includes:

- `IRMarkerTracker`: reusable YOLO detector class.
- `detect_markers`, `detect_marker_dicts`: one-frame convenience functions.
- `detect_marker_batch`, `iter_marker_detections`: multi-frame helpers.
- `load_pickle_frame`, `iter_pickle_frames`, `iter_pickle_detections`: pickle sequence helpers.
- `draw_detections`: OpenCV visualization helper.
- `BrightCircleDetector`: non-YOLO bright-circle detector for pseudo-labeling/debugging.

For production accuracy, train with human-verified labels and pass your own
`best.pt`:

```python
tracker = IRMarkerTracker("runs/detect/ir_marker_ball/weights/best.pt", device="cpu")
```

Each detection contains only the marker-ball class:

```python
[
    {
        "bbox_xyxy": [123.4, 205.1, 145.8, 228.0],
        "confidence": 0.91,
        "class_id": 0,
        "class_name": "ir_marker_ball",
        "center_xy": [134.6, 216.55],
    }
]
```

The input frame must be exactly:

- shape: `(512, 512)`
- dtype: `numpy.uint16`
- channels: one intensity channel, not RGB/BGR

## Command-Line Inference

```powershell
.\venv\Scripts\python.exe scripts\detect_npy.py frame.npy
```

## Infrared Video Example

The root-level example `example_infrared_video.py` plays `.pickle` frames from
`infrared_data` at 24 FPS and overlays marker-ball boxes and confidences. This
file is a repository example, not an installed package entry point. To use it,
clone the GitHub project and run it from the project root.

```powershell
git clone https://github.com/GGN-2015/IRYoloTracker.git
cd IRYoloTracker
.\venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run with the bundled bootstrap YOLO model:

```powershell
.\venv\Scripts\python.exe example_infrared_video.py
```

Run with custom trained weights:

```powershell
.\venv\Scripts\python.exe example_infrared_video.py --weights runs\detect\ir_marker_ball\weights\best.pt
```

Common options:

```powershell
.\venv\Scripts\python.exe example_infrared_video.py --device cuda
.\venv\Scripts\python.exe example_infrared_video.py --device cpu
.\venv\Scripts\python.exe example_infrared_video.py --fps 24 --scale 1.5
.\venv\Scripts\python.exe example_infrared_video.py --lazy-load
.\venv\Scripts\python.exe example_infrared_video.py --loop
.\venv\Scripts\python.exe example_infrared_video.py --no-rescue-bright-circles
```

At startup the example preloads all pickle frames into memory and shows a
`tqdm` progress bar. This removes disk IO and pickle decoding from the playback
loop. Use `--lazy-load` to read frames during playback instead.

The example defaults to `--conf 0.25` and enables a conservative
`YOLO+circle` rescue pass. YOLO still supplies the main predictions, but the
example also adds high-confidence bright circular blobs that do not overlap an
existing YOLO box. This helps recover marker balls that the bootstrap YOLO model
sees only at very low confidence. Use `--no-rescue-bright-circles` to view raw
YOLO output.

There is no default marker-count cap. Use `--max-detections N` only when your
camera setup has a known physical upper bound and you explicitly want to keep
the top-scoring detections.

If no YOLO weights are found, the example falls back to a simple bright-circle
preview detector so you can still inspect the infrared frame sequence. In a
normal package checkout this should not happen because a bootstrap model is
bundled. Pass `--require-yolo` if you want it to fail unless YOLO weights are
present.

Press Space to pause/resume, or press `Q`/Esc to quit.

## Bootstrap Model

The bundled model was trained from pseudo-labels generated by the preview
bright-circle detector with confidence `>= 0.70`:

```powershell
.\venv\Scripts\python.exe scripts\bootstrap_pseudo_dataset.py --confidence 0.70
.\venv\Scripts\python.exe scripts\train.py --data configs\pseudo_ir_marker_dataset.yaml --model yolo11n.pt --epochs 12 --imgsz 512 --batch 8 --device cpu --project runs\detect --name pseudo_ir_marker_bootstrap
```

This is useful as a project bootstrap. Replace it with a model trained on
human-reviewed marker-ball labels before relying on it for critical measurement.

The video example can also rebuild a lower-threshold YOLO pseudo dataset from
`infrared_data`. It uses the current inference model plus the same conservative
bright-circle rescue pass. Low-confidence YOLO boxes below
`--pseudo-yolo-accept-conf` are kept only when bright-circle evidence confirms
the same location:

```powershell
.\venv\Scripts\python.exe example_infrared_video.py --rebuild-yolo-pseudo-dataset --device cuda --pseudo-conf 0.05
```

By default this writes images and labels to
`datasets/yolo_low_conf_ir_marker_ball` and writes
`configs/yolo_low_conf_ir_marker_dataset.yaml`.

## Training Data

Prepare a YOLO detection dataset with one class:

```text
datasets/ir_marker_ball/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
```

Each label row must use normalized YOLO format:

```text
0 x_center y_center width height
```

The class id must always be `0`, named `ir_marker_ball`. Do not include labels
for non-marker objects.

If your annotations are stored with pixel-coordinate boxes, you can build the
dataset from `.npy` frames:

```powershell
.\venv\Scripts\python.exe scripts\build_dataset.py --annotations annotations.json --frames-root frames --output datasets\ir_marker_ball
```

`annotations.json` can be either a list or an object containing `frames`:

```json
[
  {
    "image": "frame_0001.npy",
    "boxes": [[120, 210, 145, 235]]
  }
]
```

## Training

The default dataset config is `configs/ir_marker_dataset.yaml`.

```powershell
.\venv\Scripts\python.exe scripts\train.py --data configs\ir_marker_dataset.yaml --model yolo11n.pt --epochs 100 --imgsz 512
```

The detector also passes `classes=[0]` during inference and filters class `0`
again after inference, so the returned output contains marker balls only.

## True One-Channel Model

For most workflows, standard YOLO training with grayscale data copied into
three equal channels is the simplest path. To use a custom YOLO model whose
first layer accepts one channel, create/train that model separately and run:

```python
tracker = IRMarkerTracker(
    "path/to/one_channel_best.pt",
    model_input_channels=1,
)
```

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest
```
