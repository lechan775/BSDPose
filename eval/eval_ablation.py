from __future__ import annotations

import argparse
import json
from pathlib import Path


ABLATION_ROWS = [
    ("yolov8_lstm", "YOLOv8-Pose", False, False, False),
    ("cbam_lstm", "YOLOv8-Pose", True, False, False),
    ("yolov8_kinematic_multitask", "YOLOv8-Pose", False, True, True),
    ("cbam_kinematic_multitask", "YOLOv8-Pose", True, True, True),
]


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_ablation_table(metrics_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for key, baseline, has_cbam, has_kinematic, has_multitask in ABLATION_ROWS:
        path = metrics_dir / key / "test_metrics.json"
        if not path.exists():
            rows.append(
                {
                    "variant": key,
                    "baseline": baseline,
                    "cbam": has_cbam,
                    "kinematic_features": has_kinematic,
                    "multi_task_head": has_multitask,
                    "missing_metrics": str(path),
                }
            )
            continue
        metrics = load_metrics(path)
        rows.append(
            {
                "variant": key,
                "baseline": baseline,
                "cbam": has_cbam,
                "kinematic_features": has_kinematic,
                "multi_task_head": has_multitask,
                "accuracy_percent": float(metrics["accuracy"]) * 100.0,
                "quality_mae": metrics.get("quality_mae"),
                "source": str(path),
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| 基线 | CBAM | 运动学特征 | 多任务头 | 分类准确率(%) | 质量MAE |",
        "|------|------|----------|---------|-------------|--------|",
    ]
    for row in rows:
        accuracy = "MISSING" if "accuracy_percent" not in row else f"{row['accuracy_percent']:.2f}"
        mae = row.get("quality_mae")
        mae_text = "—" if mae is None else f"{float(mae):.2f}"
        lines.append(
            f"| {row['baseline']} | {'✓' if row['cbam'] else ''} | "
            f"{'✓' if row['kinematic_features'] else ''} | {'✓' if row['multi_task_head'] else ''} | "
            f"{accuracy} | {mae_text} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble ablation table from classifier metric JSON files.")
    parser.add_argument("--metrics-dir", default="results/classifier")
    parser.add_argument("--output", default="results/ablation_table.json")
    parser.add_argument("--markdown-output", default="results/ablation_table.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_ablation_table(Path(args.metrics_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.markdown_output).write_text(to_markdown(rows), encoding="utf-8")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
