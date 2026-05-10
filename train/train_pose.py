from __future__ import annotations

import os

# Windows conda environments may load Intel OpenMP through both PyTorch and
# OpenCV/Albumentations. Set these before importing scientific libraries.
def _ensure_positive_int_env(name: str, default: str = "1") -> None:
    value = os.environ.get(name)
    if value is None:
        os.environ[name] = default
        return
    try:
        if int(value) <= 0:
            os.environ[name] = default
    except ValueError:
        os.environ[name] = default


_ensure_positive_int_env("OMP_NUM_THREADS")
_ensure_positive_int_env("MKL_NUM_THREADS")
_ensure_positive_int_env("OPENBLAS_NUM_THREADS")
_ensure_positive_int_env("NUMEXPR_NUM_THREADS")
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import platform
import sys
from pathlib import Path
from copy import deepcopy

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolov8_cbam_pose import build_yolov8_pose_with_attention


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_file(path: str | Path, description: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _path_exists_from_yaml(path_value: str | None, yaml_path: Path) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    if path.exists():
        return True
    if not path.is_absolute() and (yaml_path.parent / path).exists():
        return True
    return False


def make_runtime_dataset_yaml(data_yaml: Path, output_dir: Path) -> Path:
    """Create a Linux/Windows-safe dataset YAML with a valid absolute BSD root."""

    data_yaml = data_yaml.resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    original_path = payload.get("path")
    if _path_exists_from_yaml(original_path, data_yaml):
        return data_yaml

    candidate_root = data_yaml.parent
    if not (candidate_root / "images").exists() or not (candidate_root / "labels").exists():
        raise FileNotFoundError(
            f"Dataset path in {data_yaml} is invalid ({original_path!r}) and {candidate_root} "
            "does not contain images/ and labels/."
        )

    runtime_payload = deepcopy(payload)
    runtime_payload["path"] = str(candidate_root).replace("\\", "/")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = output_dir / f"{data_yaml.stem}.runtime.yaml"
    runtime_yaml.write_text(yaml.safe_dump(runtime_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return runtime_yaml


def build_model(variant: str, weights: str, reduction_ratio: int, attention_layers: list[int]):
    from ultralytics import YOLO

    if variant == "baseline":
        return YOLO(weights)
    if variant == "se":
        return build_yolov8_pose_with_attention(
            weights,
            variant="se",
            target_layers=attention_layers,
            reduction_ratio=reduction_ratio,
        )
    if variant == "cbam":
        return build_yolov8_pose_with_attention(
            weights,
            variant="cbam",
            target_layers=attention_layers,
            reduction_ratio=reduction_ratio,
        )
    raise ValueError("variant must be baseline, se, or cbam")


def _optional_arg(value, fallback):
    return fallback if value is None else value


def _parse_cache(value: str | None):
    if value is None:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def default_workers() -> int:
    return 0 if platform.system().lower() == "windows" else 8


def train_variant(variant: str, config: dict, data_yaml: Path, args: argparse.Namespace) -> dict:
    pose_cfg = config["pose"]
    logs_dir = Path(args.project or (ROOT / config["project"]["logs_dir"] / "pose"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_data_yaml = make_runtime_dataset_yaml(data_yaml, logs_dir / "_runtime_datasets")

    model = build_model(
        variant=variant,
        weights=pose_cfg["base_weights"],
        reduction_ratio=int(pose_cfg["cbam_reduction_ratio"]),
        attention_layers=[int(i) for i in pose_cfg["attention_layers"]],
    )
    train_kwargs = {
        "data": str(runtime_data_yaml),
        "imgsz": int(_optional_arg(args.imgsz, pose_cfg["image_size"])),
        "epochs": int(_optional_arg(args.epochs, pose_cfg["epochs"])),
        "batch": int(_optional_arg(args.batch, pose_cfg["batch_size"])),
        "lr0": float(_optional_arg(args.lr0, pose_cfg["lr0"])),
        "momentum": float(pose_cfg["momentum"]),
        "weight_decay": float(pose_cfg["weight_decay"]),
        "cos_lr": bool(pose_cfg["cos_lr"]),
        "optimizer": pose_cfg["optimizer"],
        "project": str(logs_dir),
        "name": f"yolov8n-pose-{variant}{args.name_suffix}",
        "exist_ok": bool(args.exist_ok),
        "workers": int(default_workers() if args.workers is None else args.workers),
        "plots": bool(args.plots),
        "patience": int(args.patience),
        "save_period": int(args.save_period),
        "fraction": float(args.fraction),
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    cache_value = _parse_cache(args.cache)
    if cache_value is not None:
        train_kwargs["cache"] = cache_value
    if args.amp is not None:
        train_kwargs["amp"] = args.amp
    if args.time is not None:
        train_kwargs["time"] = float(args.time)

    results = model.train(
        **train_kwargs,
    )

    summary = {
        "variant": variant,
        "data": str(runtime_data_yaml),
        "train_kwargs": train_kwargs,
        "save_dir": str(getattr(results, "save_dir", logs_dir / f"yolov8n-pose-{variant}{args.name_suffix}")),
    }
    out_path = logs_dir / f"{variant}_train_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8-Pose baseline/SE/CBAM variants on BSD.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--variant", choices=["baseline", "se", "cbam", "all"], default="all")
    parser.add_argument("--data", default=None, help="Ultralytics pose dataset YAML; defaults to config data.bsd_pose_yaml")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, 0,1, cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--cache", choices=["ram", "disk", "true", "false"], default=None)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patience", type=int, default=50, help="Early-stop after this many epochs without val improvement")
    parser.add_argument("--time", type=float, default=None, help="Optional wall-clock training limit in hours, passed to Ultralytics")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use, useful for pilot runs")
    parser.add_argument("--save-period", type=int, default=-1, help="Checkpoint save period in epochs; -1 keeps Ultralytics default")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_yaml = Path(args.data or config["data"]["bsd_pose_yaml"])
    require_file(data_yaml, "BSD pose dataset YAML")

    variants = config["experiments"]["pose_variants"] if args.variant == "all" else [args.variant]
    summaries = [train_variant(variant, config, data_yaml, args) for variant in variants]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
