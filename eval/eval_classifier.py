from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.dataset import KinematicFeatureDataset, STROKE_NAMES
from classifier.multi_task_head import build_classifier_model


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _torch_load(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def render_confusion_matrix(matrix: np.ndarray, labels: list[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def evaluate_checkpoint(checkpoint_path: str | Path, dataset_json: str | Path, config: dict, device: torch.device) -> dict:
    checkpoint = _torch_load(checkpoint_path, device)
    model_kind = checkpoint["model_kind"]
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    ds = KinematicFeatureDataset(
        dataset_json,
        feature_mean=None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32),
        feature_std=None if feature_std is None else np.asarray(feature_std, dtype=np.float32),
        return_sequence=(model_kind == "lstm"),
        window_size=int(config["features"]["window_size"]),
        stride=int(config["features"]["stride"]),
    )
    loader = DataLoader(ds, batch_size=int(config["classifier"]["batch_size"]), shuffle=False, num_workers=0)
    model = build_classifier_model(
        model_kind,
        num_classes=int(config["data"]["num_classes"]),
        dropout=float(config["classifier"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    score_true: list[float] = []
    score_pred: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            labels = batch["label"].to(device)
            quality = batch["quality_score"].to(device)
            if model_kind == "multi_task":
                logits, scores = model(x)
                score_true.extend(quality.cpu().numpy().tolist())
                score_pred.extend(scores.cpu().numpy().tolist())
            else:
                logits = model(x)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(int(config["data"]["num_classes"]))))
    metrics = {
        "variant": checkpoint.get("variant", Path(checkpoint_path).parent.name),
        "model_kind": model_kind,
        "num_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    if score_true:
        metrics["quality_mae"] = float(mean_absolute_error(score_true, score_pred))
        metrics["quality_rmse"] = float(mean_squared_error(score_true, score_pred) ** 0.5)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier checkpoint.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-json", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--confusion-png", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    test_json = args.test_json or config["data"]["test_json"]
    metrics = evaluate_checkpoint(args.checkpoint, test_json, config, device)

    out_path = Path(args.output or (ROOT / "results" / "classifier" / metrics["variant"] / "test_metrics.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.confusion_png:
        render_confusion_matrix(np.asarray(metrics["confusion_matrix"]), STROKE_NAMES, Path(args.confusion_png))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

