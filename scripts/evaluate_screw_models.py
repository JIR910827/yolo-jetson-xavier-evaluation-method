import argparse
import csv
import json
import time
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FAIL_WORDS = {"fail", "ng", "bad", "defect", "scratch", "thread"}


def ground_truth_from_name(path):
    name = path.name.lower()
    return "Ok" if name.startswith("ok") else "Fail"


def status_from_class(class_name, empty_status):
    if not class_name:
        return empty_status
    name = str(class_name).lower()
    return "Fail" if any(word in name for word in FAIL_WORDS) else "Ok"


def list_images(image_dir):
    return sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def best_prediction(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    names = result.names
    best = None
    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, str(cls_id))
        if best is None or conf > best[1]:
            best = (class_name, conf)
    return best


def safe_div(num, den):
    return num / den if den else 0.0


def evaluate_model(model_path, image_paths, args):
    model = YOLO(str(model_path))
    rows = []

    for idx, image_path in enumerate(image_paths, start=1):
        gt = ground_truth_from_name(image_path)

        start = time.perf_counter()
        results = model.predict(
            str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            device=args.device,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        result = results[0]
        prediction = best_prediction(result)
        if prediction is None:
            class_name = "no_detection"
            confidence = None
            pred = args.empty_status
        else:
            class_name, confidence = prediction
            pred = status_from_class(class_name, args.empty_status)

        speed = getattr(result, "speed", {}) or {}
        rows.append(
            {
                "model": model_path.name,
                "image": image_path.name,
                "ground_truth": gt,
                "prediction": pred,
                "class_name": class_name,
                "confidence": confidence,
                "preprocess_ms": speed.get("preprocess"),
                "inference_ms": speed.get("inference"),
                "postprocess_ms": speed.get("postprocess"),
                "latency_ms": latency_ms,
            }
        )

        if idx % 25 == 0:
            print(f"{model_path.name}: {idx}/{len(image_paths)} images done", flush=True)

    return rows


def summarize(rows):
    tp = sum(1 for row in rows if row["ground_truth"] == "Fail" and row["prediction"] == "Fail")
    tn = sum(1 for row in rows if row["ground_truth"] == "Ok" and row["prediction"] == "Ok")
    fp = sum(1 for row in rows if row["ground_truth"] == "Ok" and row["prediction"] == "Fail")
    fn = sum(1 for row in rows if row["ground_truth"] == "Fail" and row["prediction"] == "Ok")
    no_det = sum(1 for row in rows if row["class_name"] == "no_detection")
    total = len(rows)

    inference_values = [row["inference_ms"] for row in rows if row["inference_ms"] is not None]
    latency_values = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]

    mean_inference = sum(inference_values) / len(inference_values) if inference_values else 0.0
    mean_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0
    fps = safe_div(1000.0, mean_latency)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "model": rows[0]["model"] if rows else "",
        "total": total,
        "accuracy": safe_div(tp + tn, total),
        "precision_fail": precision,
        "recall_fail": recall,
        "f1_fail": f1,
        "tp_fail": tp,
        "tn_ok": tn,
        "fp_ok_as_fail": fp,
        "fn_fail_as_ok": fn,
        "no_detection": no_det,
        "mean_inference_ms": mean_inference,
        "mean_latency_ms": mean_latency,
        "fps": fps,
        "iou": "N/A - requires bounding-box labels",
        "map50": "N/A - requires YOLO labels",
        "map50_95": "N/A - requires YOLO labels",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="model")
    parser.add_argument("--images", default="test_images")
    parser.add_argument("--output", default="model_eval_results")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--empty-status", default="Fail", choices=["Ok", "Fail"])
    args = parser.parse_args()

    model_paths = sorted(Path(args.models).glob("*.pt"))
    image_paths = list_images(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_paths:
        raise SystemExit(f"No .pt models found in {args.models}")
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    summary_rows = []
    all_rows = []

    for model_path in model_paths:
        print(f"=== Evaluating {model_path.name} ===", flush=True)
        rows = evaluate_model(model_path, image_paths, args)
        summary = summarize(rows)
        summary_rows.append(summary)
        all_rows.extend(rows)

        pred_csv = out_dir / f"{model_path.stem}_predictions.csv"
        with pred_csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print("\nSummary:")
    for row in summary_rows:
        print(
            f"{row['model']}: acc={row['accuracy']:.4f}, "
            f"P={row['precision_fail']:.4f}, R={row['recall_fail']:.4f}, "
            f"F1={row['f1_fail']:.4f}, FPS={row['fps']:.2f}, "
            f"infer={row['mean_inference_ms']:.2f}ms, latency={row['mean_latency_ms']:.2f}ms, "
            f"no_det={row['no_detection']}"
        )
    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
