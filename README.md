# YOLO Model Evaluation Method on Jetson Xavier NX

This repository provides a reproducible method for comparing YOLO model performance on NVIDIA Jetson Xavier NX.

The main focus is model evaluation on edge hardware: detection metrics, inference speed, latency, and FPS.

## Scope

The workflow covers three tasks:

1. Verify that a YOLO model can run on Jetson Xavier NX.
2. Evaluate multiple YOLO models with common detection metrics.
3. Run batch inference on test images.

## Suggested Project Layout

Use a project directory on the Jetson device and place data under generic subdirectories:

```text
<PROJECT_DIR>/
  model/
    screw_yolov7.pt
    screw_yolov8.pt
    screw_yolov9.pt
    screw_yolov11.pt
    screw_yolov12.pt
    screw_yolov26.pt
  train/
    images/
    labels/
  test_images/
  scripts/
```

`model/` stores model weights, `train/images` and `train/labels` store YOLO-format validation data, and `test_images/` stores images used for batch inference. These directories are intentionally ignored by Git.

## Installation

### 1. Copy Scripts to the Jetson Device

```bash
cd <PROJECT_DIR>
mkdir -p scripts
cp /path/to/this-repository/scripts/* scripts/
```

When copying from another computer, use a generic SSH target:

```bash
scp -r scripts <JETSON_USER>@<JETSON_HOST>:<PROJECT_DIR>/
```

Replace `<JETSON_USER>`, `<JETSON_HOST>`, and `<PROJECT_DIR>` with the actual environment values. Do not commit those values to this repository.

### 2. Python Environment

Jetson devices should use PyTorch and torchvision builds compatible with the installed JetPack and CUDA version. Avoid replacing a working CUDA-enabled PyTorch installation with a generic CPU wheel.

Check CUDA availability:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

Install common inference and evaluation packages:

```bash
python3 -m pip install -U ultralytics pandas matplotlib scipy pyyaml pillow opencv-python
```

## YOLO26 Smoke Test

This step verifies that the model, Python environment, and CUDA runtime can complete inference on the target device.

```bash
cd <PROJECT_DIR>
bash scripts/run_yolo26_smoke_test_jetson.sh
```

## Multi-Model Evaluation

The method can be used to compare the following YOLO weight files:

```text
screw_yolov7.pt
screw_yolov8.pt
screw_yolov9.pt
screw_yolov11.pt
screw_yolov12.pt
screw_yolov26.pt
```

Create a YOLO data configuration file:

```yaml
path: <PROJECT_DIR>
train: train/images
val: train/images
names:
  0: Fail
  1: Ok
```

Run validation for Ultralytics-compatible models:

```bash
cd <PROJECT_DIR>
python3 scripts/evaluate_yolo_val_models.py \
  --data screw_eval.yaml \
  --model-dir model \
  --output-dir model_eval_results_val \
  --device 0 \
  --imgsz 640 \
  --batch 1
```

The evaluation script records:

```text
Precision
Recall
F1-score
IoU
mAP@0.5
mAP@0.5:0.95
FPS
Inference Time
Latency
```

The output directory, such as `model_eval_results_val/`, is an experimental result and should not be committed.

## Batch Inference

For image folders without YOLO labels, use the batch inference script to record predictions and timing. This is useful for practical detection speed checks when only test images are available.

```bash
cd <PROJECT_DIR>
python3 scripts/evaluate_screw_models.py \
  --model-dir model \
  --images test_images \
  --output-dir batch_inference_results \
  --device 0 \
  --imgsz 640
```

Use validation metrics when labels are available, and use batch inference timing when labels are not available.

## YOLOv7 Legacy Evaluation

Some YOLOv7 checkpoints cannot be loaded directly by the Ultralytics package because they depend on the original YOLOv7 module structure or older pickle/numpy references. In that case, evaluate YOLOv7 with the original YOLOv7 repository in a separate virtual environment.

```bash
cd <PROJECT_DIR>
sudo apt-get update
sudo apt-get install -y python3.8-venv
python3.8 -m venv --system-site-packages venv_yolov7
. venv_yolov7/bin/activate
python -m pip install "pip<25" "numpy==1.24.4"
git clone --depth 1 https://github.com/WongKinYiu/yolov7.git
```

If the original YOLOv7 code cannot read an Ultralytics-generated label cache, back it up and let YOLOv7 regenerate a compatible cache:

```bash
mv train/labels.cache train/labels.cache.ultralytics.bak
```

Then run:

```bash
cd <PROJECT_DIR>/yolov7
python test.py \
  --weights ../model/screw_yolov7.pt \
  --data ../screw_yolov7_eval.yaml \
  --img-size 640 \
  --batch-size 1 \
  --device 0 \
  --task val \
  --project ../model_eval_results_val \
  --name yolov7_legacy \
  --exist-ok \
  --no-trace
```


## Files Excluded from Version Control

The following files and directories are excluded because they may contain large binary files, experimental outputs, or environment-specific information:

```text
*.pt
*.onnx
*.engine
model/
train/
valid/
test_images/
runs/
model_eval_results*/
batch_inference_results*/
node_modules/
*.env
k.env
```

## Research Workflow Summary

This method compares YOLO model behavior on Jetson Xavier NX under the same evaluation settings. The final report should combine detection metrics, inference timing, FPS, and latency so that each YOLO version can be compared fairly on edge hardware.
