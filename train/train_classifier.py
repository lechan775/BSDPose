from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.dataset import KinematicFeatureDataset, compute_feature_normalizer, load_feature_records
from classifier.multi_task_head import MultiTaskLoss, build_classifier_model


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_epoch(model: nn.Module, loader: DataLoader, kind: str, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    abs_errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            labels = batch["label"].to(device)
            quality = batch["quality_score"].to(device)
            if kind == "multi_task":
                logits, scores = model(x)
                abs_errors.extend(torch.abs(scores - quality).cpu().tolist())
            else:
                logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    metrics = {"accuracy": correct / max(total, 1)}
    if abs_errors:
        metrics["quality_mae"] = float(np.mean(abs_errors))
    return metrics


def train_one_variant(name: str, exp_cfg: dict, config: dict, args: argparse.Namespace) -> dict:
    project_cfg = config["project"]
    classifier_cfg = config["classifier"]
    data_cfg = config["data"]
    features_cfg = config["features"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_path = Path(args.train_json or data_cfg["train_json"])
    val_path = Path(args.val_json or data_cfg["val_json"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")

    return_sequence = bool(exp_cfg["return_sequence"])
    model_kind = exp_cfg["model_kind"]
    train_records = load_feature_records(
        train_path,
        window_size=int(features_cfg["window_size"]),
        stride=int(features_cfg["stride"]),
    )
    feature_mean = feature_std = None
    if not return_sequence:
        feature_mean, feature_std = compute_feature_normalizer(train_records)

    train_ds = KinematicFeatureDataset(
        train_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        return_sequence=return_sequence,
        window_size=int(features_cfg["window_size"]),
        stride=int(features_cfg["stride"]),
    )
    val_ds = KinematicFeatureDataset(
        val_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        return_sequence=return_sequence,
        window_size=int(features_cfg["window_size"]),
        stride=int(features_cfg["stride"]),
    )

    batch_size = int(args.batch_size or classifier_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(args.workers))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(args.workers))
    model = build_classifier_model(model_kind, num_classes=int(data_cfg["num_classes"]), dropout=float(classifier_cfg["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(classifier_cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=int(classifier_cfg["reduce_lr_patience"]),
        factor=0.5,
    )
    multitask_loss = MultiTaskLoss(regression_weight=float(classifier_cfg["regression_weight"]))
    ce_loss = nn.CrossEntropyLoss()

    output_root = Path(args.output_dir or project_cfg["output_dir"])
    output_dir = (output_root if output_root.is_absolute() else ROOT / output_root) / "classifier" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_epoch = -1
    patience_left = int(classifier_cfg["early_stopping_patience"])
    history: list[dict] = []

    epochs = int(args.epochs or classifier_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch["features"].to(device)
            labels = batch["label"].to(device)
            quality = batch["quality_score"].to(device)
            optimizer.zero_grad(set_to_none=True)
            if model_kind == "multi_task":
                logits, scores = model(x)
                loss = multitask_loss(logits, scores, labels, quality)
            else:
                logits = model(x)
                loss = ce_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())

        val_metrics = evaluate_epoch(model, val_loader, model_kind, device)
        scheduler.step(val_metrics["accuracy"])
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_ds), 1),
            **val_metrics,
        }
        history.append(row)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            patience_left = int(classifier_cfg["early_stopping_patience"])
            checkpoint = {
                "variant": name,
                "model_kind": model_kind,
                "state_dict": model.state_dict(),
                "feature_mean": None if feature_mean is None else feature_mean.tolist(),
                "feature_std": None if feature_std is None else feature_std.tolist(),
                "config": config,
                "best_epoch": best_epoch,
                "best_val_metrics": val_metrics,
            }
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    summary = {
        "variant": name,
        "model_kind": model_kind,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "checkpoint": str(output_dir / "best.pt"),
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train badminton stroke classifiers.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--variant", default="all", help="Experiment key from config, or all")
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--val-json", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    experiments = config["experiments"]["classifier_variants"]
    names = list(experiments) if args.variant == "all" else [args.variant]
    summaries = []
    for name in names:
        if name not in experiments:
            raise KeyError(f"Unknown classifier variant: {name}")
        summaries.append(train_one_variant(name, experiments[name], config, args))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
