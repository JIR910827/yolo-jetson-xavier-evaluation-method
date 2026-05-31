#!/usr/bin/env python3
"""Evaluate multiple Ultralytics-compatible YOLO models on a Jetson device."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_speed_ms(results: Any) -> tuple[float | None, float | None]:
    speed = getattr(results, "speed", {}) or {}
    inference = scalar(speed.get("inference"))
    latency_parts = [
        scalar(speed.get("preprocess")),
        inference,
        scalar(speed.get("postprocess")),
    ]
    latency = sum(part for part in latency_parts if part is not None)
    return inference, latency


def evaluate_model(model_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    model = YOLO(str(model_path))
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        project=str(args.output_dir),
        name=model_path.stem,
        exist_ok=True,
        verbose=False,
    )

    box = results.box
    precision = scalar(box.mp)
    recall = scalar(box.mr)
    map50 = scalar(box.map50)
    map5095 = scalar(box.map)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    inference_ms, latency_ms = get_speed_ms(results)
    fps = 1000 / latency_ms if latency_ms and latency_ms > 0 else None

    return {
        "model": model_path.name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": map5095,
        "map50": map50,
        "map50_95": map5095,
        "fps": fps,
        "inference_time_ms": inference_ms,
        "latency_ms": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO data yaml path")
    parser.add_argument("--model-dir", default="model", type=Path)
    parser.add_argument("--output-dir", default="model_eval_results_val", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--pattern",
        default="screw_yolov*.pt",
        help="Glob pattern for model weights. YOLOv7 legacy weights may require the original YOLOv7 repo.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(args.model_dir.glob(args.pattern))
    if not model_paths:
        raise FileNotFoundError(f"No model files matched {args.model_dir / args.pattern}")

    rows: list[dict[str, Any]] = []
    for model_path in model_paths:
        try:
            row = evaluate_model(model_path, args)
            row["status"] = "ok"
        except Exception as exc:  # keep batch evaluation moving
            row = {"model": model_path.name, "status": "failed", "error": str(exc)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    csv_path = args.output_dir / "summary_val.csv"
    json_path = args.output_dir / "summary_val.json"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
