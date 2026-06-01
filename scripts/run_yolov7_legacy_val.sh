#!/usr/bin/env bash
set -euo pipefail

cd "${YOLO_EVAL_DIR:-$PWD}"

if [ ! -d "venv_yolov7" ]; then
  python3.8 -m venv --system-site-packages venv_yolov7
fi

. venv_yolov7/bin/activate
python -m pip install "pip<25" "numpy==1.24.4"

if [ ! -d "yolov7" ]; then
  git clone --depth 1 https://github.com/WongKinYiu/yolov7.git
fi

if [ -f "train/labels.cache" ] && [ ! -f "train/labels.cache.ultralytics.bak" ]; then
  mv train/labels.cache train/labels.cache.ultralytics.bak
fi

cd yolov7
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
