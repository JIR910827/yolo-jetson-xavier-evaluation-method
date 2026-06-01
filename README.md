# YOLO Model Evaluation Method on Jetson Xavier NX

This repository provides a reproducible method for evaluating screw defect detection models on NVIDIA Jetson Xavier NX and, when required, uploading recognition metadata to a private Ethereum smart contract.

The repository is designed as a method package. It does not include model weights, image datasets, validation outputs, private keys, local environment files, or device-specific addresses.

## Scope

The workflow covers four tasks:

1. Verify that a YOLO model can run on Jetson Xavier NX.
2. Evaluate multiple YOLO models with common detection metrics.
3. Run batch inference on screw images.
4. Upload image hash and recognition metadata to an Ethereum private chain.

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
  VisualRecordContract.abi.json
  VisualRecordContract.address.txt
```

`model/` stores model weights, `train/images` and `train/labels` store YOLO-format validation data, and `test_images/` stores images used for batch recognition. These directories are intentionally ignored by Git.

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

### 3. Node.js and Ethereum Contract Access

The upload and query scripts use `web3@1.10.0`.

```bash
cd <PROJECT_DIR>
npm init -y
npm install web3@1.10.0
```

The Ethereum private-chain RPC endpoint and contract files are configured through environment variables or local files:

```bash
export ETH_RPC_URL="http://127.0.0.1:8545"
export CONTRACT_ABI_PATH="VisualRecordContract.abi.json"
export CONTRACT_ADDRESS_PATH="VisualRecordContract.address.txt"
export DEVICE_ID="jetson-xavier-nx-0"
```

If the legacy HTTP API mode is used, configure the API URL explicitly:

```bash
export BLOCKCHAIN_API_URL="http://<BLOCKCHAIN_API_HOST>:3000/iotBlockChain/CreateVisualRecord"
```

## Script Usage

### 1. YOLO26 Smoke Test

This step verifies that the model, Python environment, and CUDA runtime can complete inference on the target device.

```bash
cd <PROJECT_DIR>
bash scripts/run_yolo26_smoke_test_jetson.sh
```

### 2. Batch Recognition and On-Chain Upload

This command reads images from `test_images/`, performs YOLO inference, computes SHA-256 image hashes, and writes visual records to the smart contract.

```bash
cd <PROJECT_DIR>
python3 scripts/batch_detect_screw_upload.py \
  --model model/screw_yolov26.pt \
  --images test_images \
  --device 0 \
  --empty-status Fail
```

The uploaded `Result` field is stored as a JSON string. The main fields are:

```text
product_type
status
class_name
is_defect
confidence
boxes
```

### 3. Query On-Chain Visual Records

Query all records:

```bash
cd <PROJECT_DIR>
node scripts/query_visual_records.js
```

Query one record:

```bash
node scripts/query_visual_records.js --id <RECORD_ID>
```

### 4. Upload One Recognition Record

```bash
cd <PROJECT_DIR>
node scripts/upload_recognition_to_chain.js \
  --id screw-test-001 \
  --datetime "2026-05-31 20:00:00" \
  --image test_images/sample.jpg \
  --status Ok \
  --class-name Ok \
  --confidence 0.93 \
  --device-id jetson-xavier-nx-0
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



## Research Workflow Summary

The method connects AOI recognition and blockchain-based traceability. A YOLO model first performs screw defect detection on Jetson Xavier NX. The system then calculates an image hash and combines it with recognition metadata, confidence scores, device ID, and timestamps. Finally, the metadata is written to an Ethereum private-chain smart contract so that inspection records can be queried and verified later.
