import platform
import time

import numpy as np


def main():
    print("=== YOLO26 Jetson smoke test ===")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"[FAIL] Cannot import torch: {exc}") from exc

    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        import ultralytics
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"[FAIL] Cannot import ultralytics: {exc}") from exc

    print(f"Ultralytics: {ultralytics.__version__}")

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[140:500, 220:430] = (255, 255, 255)

    print("Loading yolo26n.pt ...")
    load_start = time.perf_counter()
    try:
        model = YOLO("yolo26n.pt")
    except Exception as exc:
        raise SystemExit(f"[FAIL] Cannot load yolo26n.pt: {exc}") from exc
    print(f"Model loaded in {(time.perf_counter() - load_start):.2f}s")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running inference on device={device} ...")
    infer_start = time.perf_counter()
    try:
        results = model.predict(image, imgsz=640, device=device, verbose=False)
    except Exception as exc:
        raise SystemExit(f"[FAIL] Inference failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - infer_start) * 1000
    boxes = len(results[0].boxes) if results and results[0].boxes is not None else 0
    print(f"[OK] Inference completed in {elapsed_ms:.1f} ms")
    print(f"Detections: {boxes}")


if __name__ == "__main__":
    main()
