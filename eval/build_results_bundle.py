from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.dataset import STROKE_NAMES


TABLE3_SOURCES = [
    ("OpenPose + LSTM", "openpose_lstm"),
    ("YOLOv8-Pose + LSTM", "yolov8_lstm"),
    ("YOLOv8-Pose + 运动学特征 + MLP", "yolov8_kinematic_mlp"),
    ("本文方法(改进YOLOv8 + 运动学特征 + 多任务头)", "cbam_kinematic_multitask"),
]

TABLE5_SOURCES = [
    ("YOLOv8-Pose", False, False, False, "yolov8_lstm"),
    ("YOLOv8-Pose", True, False, False, "cbam_lstm"),
    ("YOLOv8-Pose", False, True, True, "yolov8_kinematic_multitask"),
    ("YOLOv8-Pose", True, True, True, "cbam_kinematic_multitask"),
]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def percent(value: float) -> float:
    return float(value) * 100.0


def build_table3(classifier_dir: Path) -> list[dict]:
    rows = []
    for method, key in TABLE3_SOURCES:
        metrics = read_json(classifier_dir / key / "test_metrics.json")
        rows.append(
            {
                "method": method,
                "accuracy": percent(metrics["accuracy"]),
                "precision_weighted": percent(metrics["precision_weighted"]),
                "recall_weighted": percent(metrics["recall_weighted"]),
                "f1_weighted": percent(metrics["f1_weighted"]),
                "source": str(classifier_dir / key / "test_metrics.json"),
            }
        )
    return rows


def build_table4(classifier_dir: Path) -> dict:
    metrics = read_json(classifier_dir / "cbam_kinematic_multitask" / "test_metrics.json")
    return {"labels": STROKE_NAMES, "matrix": metrics["confusion_matrix"]}


def build_table5(classifier_dir: Path) -> list[dict]:
    rows = []
    for baseline, cbam, kinematic, multitask, key in TABLE5_SOURCES:
        metrics = read_json(classifier_dir / key / "test_metrics.json")
        rows.append(
            {
                "baseline": baseline,
                "cbam": cbam,
                "kinematic_features": kinematic,
                "multi_task_head": multitask,
                "accuracy": percent(metrics["accuracy"]),
                "quality_mae": metrics.get("quality_mae"),
                "source": str(classifier_dir / key / "test_metrics.json"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper result bundle from real metric JSON files.")
    parser.add_argument("--pose-table2", required=True, help="JSON list with table2 rows")
    parser.add_argument("--classifier-dir", default="results/classifier")
    parser.add_argument("--output", default="results/paper_results_bundle.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = {
        "table2": read_json(Path(args.pose_table2)),
        "table3": build_table3(Path(args.classifier_dir)),
        "table4": build_table4(Path(args.classifier_dir)),
        "table5": build_table5(Path(args.classifier_dir)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(bundle, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
