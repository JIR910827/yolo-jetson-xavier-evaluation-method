import argparse
import json
import subprocess
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FAIL_WORDS = {"fail", "ng", "bad", "defect", "defective"}


def normalize_status(class_name, empty_status):
    name = str(class_name or "").strip().lower()
    if not name:
        return empty_status
    return "Fail" if any(word in name for word in FAIL_WORDS) else "Ok"


def list_images(image_dir):
    return sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def best_detection(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, []

    names = result.names
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = [round(float(value), 3) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(confidence, 6),
                "xyxy": xyxy,
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections[0], detections


def upload_record(args, image_path, status, class_name, confidence, boxes):
    record_id = f"screw-{image_path.stem}"
    command = [
        "node",
        str(args.uploader),
        "--mode",
        args.mode,
        "--status",
        status,
        "--id",
        record_id,
        "--pic-id",
        record_id,
        "--image",
        str(image_path),
        "--device",
        args.device_id,
        "--class-name",
        class_name,
        "--boxes",
        json.dumps(boxes, separators=(",", ":")),
    ]

    if confidence is not None:
        command.extend(["--confidence", str(confidence)])
    if args.rpc_url:
        command.extend(["--rpc-url", args.rpc_url])
    if args.abi_path:
        command.extend(["--abi-path", args.abi_path])
    if args.address_path:
        command.extend(["--address-path", args.address_path])
    if args.dry_run:
        print("[DRY RUN]", " ".join(command))
        return 0

    completed = subprocess.run(command, text=True, capture_output=True)
    print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Detect screw Ok/Fail images and upload records to chain."
    )
    parser.add_argument("--model", required=True, help="Path to YOLO best.pt")
    parser.add_argument("--images", required=True, help="Folder containing test images")
    parser.add_argument(
        "--uploader",
        default="./upload_recognition_to_chain.js",
        help="Path to upload_recognition_to_chain.js",
    )
    parser.add_argument("--mode", default="contract", choices=["contract", "api"])
    parser.add_argument("--device-id", default="jetson-xavier-nx-0")
    parser.add_argument("--rpc-url", default="")
    parser.add_argument("--abi-path", default="")
    parser.add_argument("--address-path", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--empty-status",
        default="Fail",
        choices=["Ok", "Fail"],
        help="Status to upload when YOLO returns no detections.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    image_paths = list_images(args.images)
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    model = YOLO(args.model)
    success = 0
    failed = 0

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{index}/{len(image_paths)}] {image_path.name}")
        results = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
        best, boxes = best_detection(results[0])

        if best is None:
            status = args.empty_status
            class_name = "no_detection"
            confidence = None
        else:
            status = normalize_status(best["class_name"], args.empty_status)
            class_name = best["class_name"]
            confidence = best["confidence"]

        print(f"Detect result: status={status}, class={class_name}, confidence={confidence}")
        code = upload_record(args, image_path, status, class_name, confidence, boxes)
        if code == 0:
            success += 1
        else:
            failed += 1
            print(f"Upload failed with exit code {code}")

    print(f"\nDone. success={success}, failed={failed}, total={len(image_paths)}")


if __name__ == "__main__":
    main()
