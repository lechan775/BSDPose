from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COCO_SIGMAS = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
    dtype=np.float32,
) / 10.0


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _reshape_keypoints(keypoints: Any) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        elif arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[0] < 17 or arr.shape[1] < 2:
        raise ValueError("keypoints must describe at least 17 points")
    if arr.shape[1] == 2:
        visibility = np.ones((arr.shape[0], 1), dtype=np.float32)
        arr = np.concatenate([arr, visibility], axis=1)
    return arr[:17, :3]


def _records(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if key in data:
        return data[key]
    if "annotations" in data:
        return data["annotations"]
    if "predictions" in data:
        return data["predictions"]
    raise ValueError(f"Could not find records in JSON with key {key}")


def _bbox_area(record: dict[str, Any], keypoints: np.ndarray) -> float:
    if "area" in record and record["area"]:
        return max(float(record["area"]), 1.0)
    if "bbox" in record:
        bbox = record["bbox"]
        if len(bbox) >= 4:
            return max(float(bbox[2]) * float(bbox[3]), 1.0)
    visible = keypoints[keypoints[:, 2] > 0]
    if len(visible) == 0:
        visible = keypoints
    width = float(np.nanmax(visible[:, 0]) - np.nanmin(visible[:, 0]))
    height = float(np.nanmax(visible[:, 1]) - np.nanmin(visible[:, 1]))
    return max(width * height, 1.0)


def oks(gt_kpts: np.ndarray, pred_kpts: np.ndarray, area: float) -> float:
    visible = gt_kpts[:, 2] > 0
    if not np.any(visible):
        visible = np.ones(gt_kpts.shape[0], dtype=bool)
    diff = gt_kpts[:, :2] - pred_kpts[:, :2]
    squared_distance = np.sum(diff * diff, axis=1)
    variances = (COCO_SIGMAS * 2.0) ** 2
    e = squared_distance / (2.0 * area * variances + np.spacing(1))
    return float(np.mean(np.exp(-e[visible])))


def pck_at_torso(ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold_ratio: float = 0.2) -> float:
    pred_by_image = {str(item.get("image_id", item.get("id"))): item for item in predictions}
    correct = 0
    total = 0
    for gt in ground_truth:
        key = str(gt.get("image_id", gt.get("id")))
        if key not in pred_by_image:
            continue
        gt_kpts = _reshape_keypoints(gt["keypoints"])
        pred_kpts = _reshape_keypoints(pred_by_image[key]["keypoints"])
        shoulder_mid = (gt_kpts[5, :2] + gt_kpts[6, :2]) / 2.0
        hip_mid = (gt_kpts[11, :2] + gt_kpts[12, :2]) / 2.0
        torso = float(np.linalg.norm(shoulder_mid - hip_mid))
        if torso <= 1e-6:
            continue
        threshold = threshold_ratio * torso
        visible = gt_kpts[:, 2] > 0
        distances = np.linalg.norm(gt_kpts[:, :2] - pred_kpts[:, :2], axis=1)
        correct += int(np.sum(distances[visible] <= threshold))
        total += int(np.sum(visible))
    return 100.0 * correct / max(total, 1)


def oks_map(ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> float:
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    for gt in ground_truth:
        gt_by_image.setdefault(str(gt.get("image_id", gt.get("id"))), []).append(gt)

    pred_sorted = sorted(predictions, key=lambda item: float(item.get("score", 1.0)), reverse=True)
    aps: list[float] = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        matched: set[tuple[str, int]] = set()
        tp: list[int] = []
        fp: list[int] = []
        for pred in pred_sorted:
            image_key = str(pred.get("image_id", pred.get("id")))
            candidates = gt_by_image.get(image_key, [])
            pred_kpts = _reshape_keypoints(pred["keypoints"])
            best_score = -1.0
            best_index = -1
            for idx, gt in enumerate(candidates):
                if (image_key, idx) in matched:
                    continue
                gt_kpts = _reshape_keypoints(gt["keypoints"])
                score = oks(gt_kpts, pred_kpts, _bbox_area(gt, gt_kpts))
                if score > best_score:
                    best_score = score
                    best_index = idx
            if best_score >= threshold and best_index >= 0:
                matched.add((image_key, best_index))
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        if not tp:
            aps.append(0.0)
            continue
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(sum(len(v) for v in gt_by_image.values()), 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        ap = 0.0
        for recall_level in np.linspace(0.0, 1.0, 101):
            precision_at_recall = precisions[recalls >= recall_level]
            ap += float(np.max(precision_at_recall)) if precision_at_recall.size else 0.0
        aps.append(ap / 101.0)
    return 100.0 * float(np.mean(aps))


def evaluate_bsd_predictions(ground_truth_json: str | Path, predictions_json: str | Path) -> dict[str, float]:
    gt_records = _records(_load_json(ground_truth_json), "annotations")
    pred_records = _records(_load_json(predictions_json), "predictions")
    return {
        "bsd_map": oks_map(gt_records, pred_records),
        "bsd_pck_0_2": pck_at_torso(gt_records, pred_records, threshold_ratio=0.2),
        "num_ground_truth": len(gt_records),
        "num_predictions": len(pred_records),
    }


def evaluate_ultralytics(model_path: str | Path, data_yaml: str | Path, image_size: int, split: str = "val") -> dict[str, float]:
    from ultralytics import YOLO

    metrics = YOLO(str(model_path)).val(data=str(data_yaml), imgsz=image_size, split=split)
    pose_metrics = getattr(metrics, "pose", None)
    box_metrics = getattr(metrics, "box", None)
    output: dict[str, float] = {}
    if pose_metrics is not None and hasattr(pose_metrics, "map"):
        output["pose_map"] = float(pose_metrics.map) * 100.0
    elif box_metrics is not None and hasattr(box_metrics, "map"):
        output["box_map"] = float(box_metrics.map) * 100.0
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-Pose variants.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--model", default=None, help="YOLO .pt checkpoint for Ultralytics val")
    parser.add_argument("--data-yaml", default=None, help="Dataset YAML for Ultralytics val")
    parser.add_argument("--ground-truth", default=None, help="BSD test JSON with keypoints")
    parser.add_argument("--predictions", default=None, help="Prediction JSON with keypoints")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", default=str(ROOT / "results" / "pose_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics: dict[str, float] = {}
    if args.model and args.data_yaml:
        metrics.update(evaluate_ultralytics(args.model, args.data_yaml, int(config["pose"]["image_size"]), split=args.split))
    if args.ground_truth and args.predictions:
        metrics.update(evaluate_bsd_predictions(args.ground_truth, args.predictions))
    if not metrics:
        raise ValueError("Provide --model and --data-yaml, or --ground-truth and --predictions")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
