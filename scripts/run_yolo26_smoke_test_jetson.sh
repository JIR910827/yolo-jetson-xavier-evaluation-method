#!/bin/bash
set -euo pipefail

echo "=== Environment ==="
python3 -V
python3 -m pip show ultralytics torch torchvision || true

echo ""
echo "=== Optional GPU status ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

echo ""
echo "=== YOLO26 smoke test ==="
python3 test_yolo26_jetson.py
