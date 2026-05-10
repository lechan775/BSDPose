from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.kinematic import aggregate_window_features


STROKE_NAMES = [
    "forehand_clear",
    "backhand_clear",
    "forehand_smash",
    "backhand_smash",
    "forehand_net_lift",
    "backhand_net_lift",
]

FAMILY_TO_LABEL = {
    ("clear", "forehand"): 0,
    ("clear", "backhand"): 1,
    ("smash", "forehand"): 2,
    ("smash", "backhand"): 3,
    ("net_lift", "forehand"): 4,
    ("net_lift", "backhand"): 5,
}

VIDEOBADMINTON_CLASS_TO_FAMILY = {
    "02_Lift": "net_lift",
    "10_Defensive Clear": "clear",
    "12_Clear": "clear",
    "03_Tap Smash": "smash",
    "14_Smash": "smash",
}

GENERATED_PATHS = [
    "images",
    "labels",
    "train.json",
    "val.json",
    "test.json",
    "bsd-pose.yaml",
    "conversion_manifest.json",
]


@dataclass
class FramePose:
    keypoints: np.ndarray
    bbox_xyxy: np.ndarray
    score: float
    detected: bool


@dataclass
class SampleRecord:
    sample_id: str
    source_video: str
    source_class: str
    stroke_family: str
    split: str
    frame_paths: list[str]
    keypoints_sequence: np.ndarray
    bbox_sequence: np.ndarray
    detection_rate: float
    confidence_score: float
    smoothness_score: float
    features: np.ndarray
    handedness: str = ""
    stroke_type: int = -1
    stroke_name: str = ""
    quality_score: float = 0.0
    template_score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_record(self, output_root: Path) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_video": self.source_video,
            "source_class": self.source_class,
            "stroke_family": self.stroke_family,
            "split": self.split,
            "frame_paths": [str(Path(p).relative_to(output_root)).replace("\\", "/") for p in self.frame_paths],
            "keypoints_sequence": self.keypoints_sequence.round(4).tolist(),
            "bbox_sequence": self.bbox_sequence.round(4).tolist(),
            "features": self.features.round(6).tolist(),
            "stroke_type": int(self.stroke_type),
            "stroke_name": self.stroke_name,
            "stroke_type_source": "VideoBadminton class family + COCO pose-side forehand/backhand heuristic",
            "handedness": self.handedness,
            "quality_score": round(float(self.quality_score), 4),
            "quality_source": "pseudo: 30% keypoint confidence + 40% class-template similarity + 30% wrist-trajectory smoothness",
            "detection_rate": round(float(self.detection_rate), 4),
            "confidence_score": round(float(self.confidence_score), 4),
            "template_score": round(float(self.template_score), 4),
            "smoothness_score": round(float(self.smoothness_score), 4),
            **self.meta,
        }


def sanitize(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_")[:120] or "sample"


def stable_id(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{sanitize(path.parent.name)}_{sanitize(path.stem)}_{digest}"


def assert_inside(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Refusing to modify outside output root: {resolved}")


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [output_root / rel for rel in GENERATED_PATHS if (output_root / rel).exists()]
    if existing and not overwrite:
        joined = "\n".join(str(p) for p in existing)
        raise FileExistsError(
            "Generated BSD outputs already exist. Re-run with --overwrite to replace them:\n" + joined
        )
    if overwrite:
        for rel in GENERATED_PATHS:
            target = output_root / rel
            if not target.exists():
                continue
            assert_inside(target, output_root)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_videos(source_root: Path, limit_per_source_class: int | None) -> list[tuple[Path, str, str]]:
    video_root = source_root / "VideoBadminton_Dataset"
    if not video_root.exists():
        raise FileNotFoundError(f"VideoBadminton_Dataset not found: {video_root}")

    rows: list[tuple[Path, str, str]] = []
    for class_name, family in VIDEOBADMINTON_CLASS_TO_FAMILY.items():
        class_dir = video_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Required VideoBadminton class missing: {class_dir}")
        videos = sorted(class_dir.rglob("*.mp4"))
        if limit_per_source_class is not None:
            videos = videos[:limit_per_source_class]
        rows.extend((path, class_name, family) for path in videos)
    if not rows:
        raise ValueError(f"No mapped VideoBadminton .mp4 files found under {video_root}")
    return rows


def assign_splits(
    videos: list[tuple[Path, str, str]],
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> dict[Path, str]:
    rng = random.Random(seed)
    grouped: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for row in videos:
        grouped[row[2]].append(row)

    splits: dict[Path, str] = {}
    for rows in grouped.values():
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        for idx, (path, _, _) in enumerate(rows):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            splits[path] = split
    return splits


def frame_indices(frame_count: int, window_size: int, mode: str) -> list[int]:
    if frame_count <= 0:
        return []
    if mode == "linspace":
        return np.linspace(0, frame_count - 1, window_size).round().astype(int).tolist()
    if frame_count >= window_size:
        start = max(0, (frame_count - window_size) // 2)
        return list(range(start, start + window_size))
    return np.linspace(0, frame_count - 1, window_size).round().astype(int).tolist()


def read_video_window(video_path: Path, window_size: int, sampling: str) -> tuple[list[np.ndarray], list[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = frame_indices(total, window_size, sampling)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            if frames:
                frame = frames[-1].copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    cap.release()
    if len(frames) != window_size:
        raise RuntimeError(f"Expected {window_size} frames from {video_path}, got {len(frames)}")
    return frames, indices


def choose_person(result: Any) -> FramePose:
    empty_kpts = np.zeros((17, 3), dtype=np.float32)
    empty_bbox = np.zeros(5, dtype=np.float32)
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return FramePose(empty_kpts, empty_bbox, 0.0, False)

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    box_conf = result.boxes.conf.detach().cpu().numpy()
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas * np.maximum(box_conf, 1e-6)))

    xy = result.keypoints.xy[idx].detach().cpu().numpy()
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[idx].detach().cpu().numpy()
    else:
        conf = np.ones((xy.shape[0],), dtype=np.float32)
    keypoints = np.concatenate([xy[:17], conf[:17, None]], axis=1).astype(np.float32)
    bbox = np.array([boxes[idx, 0], boxes[idx, 1], boxes[idx, 2], boxes[idx, 3], box_conf[idx]], dtype=np.float32)
    return FramePose(keypoints, bbox, float(box_conf[idx]), True)


def infer_pose_sequence(model: Any, frames: list[np.ndarray], imgsz: int, conf: float, device: str) -> list[FramePose]:
    results = model.predict(frames, imgsz=imgsz, conf=conf, verbose=False, device=device)
    poses = [choose_person(result) for result in results]
    last_keypoints: np.ndarray | None = None
    last_bbox: np.ndarray | None = None
    for pose in poses:
        if pose.detected:
            last_keypoints = pose.keypoints.copy()
            last_bbox = pose.bbox_xyxy.copy()
        elif last_keypoints is not None and last_bbox is not None:
            pose.keypoints = last_keypoints.copy()
            pose.keypoints[:, 2] = 0.0
            pose.bbox_xyxy = last_bbox.copy()
            pose.bbox_xyxy[4] = 0.0
    return poses


def bbox_from_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    valid = keypoints[keypoints[:, 2] > 0.0]
    if len(valid) == 0:
        valid = keypoints
    x1 = float(np.clip(np.min(valid[:, 0]), 0, width - 1))
    y1 = float(np.clip(np.min(valid[:, 1]), 0, height - 1))
    x2 = float(np.clip(np.max(valid[:, 0]), 0, width - 1))
    y2 = float(np.clip(np.max(valid[:, 1]), 0, height - 1))
    pad_x = max(4.0, 0.08 * (x2 - x1 + 1.0))
    pad_y = max(4.0, 0.08 * (y2 - y1 + 1.0))
    return np.array(
        [
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(width - 1), x2 + pad_x),
            min(float(height - 1), y2 + pad_y),
            0.0,
        ],
        dtype=np.float32,
    )


def write_yolo_pose_label(label_path: Path, keypoints: np.ndarray, bbox: np.ndarray, width: int, height: int) -> None:
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        bbox = bbox_from_keypoints(keypoints, width, height)
    x_center = ((bbox[0] + bbox[2]) / 2.0) / width
    y_center = ((bbox[1] + bbox[3]) / 2.0) / height
    box_w = (bbox[2] - bbox[0]) / width
    box_h = (bbox[3] - bbox[1]) / height
    values = [0, x_center, y_center, box_w, box_h]
    for x, y, kconf in keypoints:
        visibility = 2.0 if kconf >= 0.25 else 1.0
        values.extend([float(x) / width, float(y) / height, visibility])
    label_path.write_text(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values) + "\n", encoding="utf-8")


def infer_handedness(keypoints_sequence: np.ndarray) -> str:
    right = keypoints_sequence[:, 10, :2]
    left = keypoints_sequence[:, 9, :2]
    right_speed = np.linalg.norm(np.diff(right, axis=0), axis=1)
    left_speed = np.linalg.norm(np.diff(left, axis=0), axis=1)
    if float(np.nanmax(right_speed)) >= float(np.nanmax(left_speed)):
        active = "right"
        wrist = right
        peak = int(np.argmax(np.r_[0.0, right_speed]))
    else:
        active = "left"
        wrist = left
        peak = int(np.argmax(np.r_[0.0, left_speed]))

    torso_center_x = float(
        np.nanmean(
            [
                keypoints_sequence[peak, 5, 0],
                keypoints_sequence[peak, 6, 0],
                keypoints_sequence[peak, 11, 0],
                keypoints_sequence[peak, 12, 0],
            ]
        )
    )
    wrist_x = float(wrist[peak, 0])
    if active == "right":
        return "forehand" if wrist_x >= torso_center_x else "backhand"
    return "forehand" if wrist_x <= torso_center_x else "backhand"


def confidence_score(keypoints_sequence: np.ndarray) -> float:
    return float(np.clip(np.nanmean(keypoints_sequence[:, :, 2]) * 100.0, 0.0, 100.0))


def smoothness_score(keypoints_sequence: np.ndarray) -> float:
    right_speed = np.linalg.norm(np.diff(keypoints_sequence[:, 10, :2], axis=0), axis=1)
    left_speed = np.linalg.norm(np.diff(keypoints_sequence[:, 9, :2], axis=0), axis=1)
    speed = right_speed if np.nanmax(right_speed) >= np.nanmax(left_speed) else left_speed
    if len(speed) < 2:
        return 50.0
    acceleration = np.diff(speed)
    ratio = float(np.nanstd(acceleration) / max(np.nanmean(speed), 1.0))
    return float(np.clip(100.0 * math.exp(-ratio), 0.0, 100.0))


def assign_template_and_quality(records: list[SampleRecord]) -> None:
    if not records:
        return
    feature_matrix = np.stack([record.features for record in records], axis=0)
    global_std = np.maximum(np.nanstd(feature_matrix, axis=0), 1e-6)
    by_label: dict[int, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_label[record.stroke_type].append(record)

    for label_records in by_label.values():
        matrix = np.stack([record.features for record in label_records], axis=0)
        template = np.nanmedian(matrix, axis=0)
        distances = np.linalg.norm((matrix - template) / global_std, axis=1)
        scale = max(float(np.nanmedian(distances)), 1.0)
        for record, distance in zip(label_records, distances):
            template_score = 100.0 * math.exp(-float(distance) / scale)
            record.template_score = float(np.clip(template_score, 0.0, 100.0))
            record.quality_score = float(
                np.clip(
                    0.30 * record.confidence_score + 0.40 * record.template_score + 0.30 * record.smoothness_score,
                    0.0,
                    100.0,
                )
            )


def process_video(
    video_path: Path,
    source_class: str,
    family: str,
    split: str,
    output_root: Path,
    model: Any,
    args: argparse.Namespace,
) -> SampleRecord | None:
    frames, indices = read_video_window(video_path, args.window_size, args.sampling)
    poses = infer_pose_sequence(model, frames, imgsz=args.imgsz, conf=args.pose_conf, device=args.device)
    detected_count = sum(1 for pose in poses if pose.detected)
    detection_rate = detected_count / max(len(poses), 1)
    if detected_count < args.min_detected_frames:
        return None

    sample_id = stable_id(video_path)
    frame_paths: list[str] = []
    keypoints_sequence = np.stack([pose.keypoints for pose in poses], axis=0).astype(np.float32)
    bbox_sequence = np.stack([pose.bbox_xyxy for pose in poses], axis=0).astype(np.float32)

    for frame_idx, (frame, pose, source_frame_idx) in enumerate(zip(frames, poses, indices)):
        image_name = f"{sample_id}_{frame_idx:02d}.jpg"
        label_name = f"{sample_id}_{frame_idx:02d}.txt"
        image_path = output_root / "images" / split / image_name
        label_path = output_root / "labels" / split / label_name
        height, width = frame.shape[:2]
        cv2.imwrite(str(image_path), frame)
        write_yolo_pose_label(label_path, pose.keypoints, pose.bbox_xyxy, width, height)
        frame_paths.append(str(image_path))

    features = aggregate_window_features(keypoints_sequence)
    handedness = infer_handedness(keypoints_sequence)
    stroke_type = FAMILY_TO_LABEL[(family, handedness)]
    return SampleRecord(
        sample_id=sample_id,
        source_video=str(video_path),
        source_class=source_class,
        stroke_family=family,
        split=split,
        frame_paths=frame_paths,
        keypoints_sequence=keypoints_sequence,
        bbox_sequence=bbox_sequence,
        detection_rate=detection_rate,
        confidence_score=confidence_score(keypoints_sequence),
        smoothness_score=smoothness_score(keypoints_sequence),
        features=features,
        handedness=handedness,
        stroke_type=stroke_type,
        stroke_name=STROKE_NAMES[stroke_type],
        meta={"source_frame_indices": [int(i) for i in indices]},
    )


def write_split_json(output_root: Path, split: str, records: list[SampleRecord], metadata: dict[str, Any]) -> None:
    payload = {
        "metadata": metadata | {"split": split, "num_samples": len(records)},
        "samples": [record.to_json_record(output_root) for record in records],
    }
    (output_root / f"{split}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_bsd_pose_yaml(output_root: Path) -> None:
    payload = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [17, 3],
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        "names": {0: "person"},
        "kpt_names": {
            0: [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ]
        },
    }
    (output_root / "bsd-pose.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def make_manifest(output_root: Path, source_root: Path, records: list[SampleRecord], args: argparse.Namespace) -> dict[str, Any]:
    by_split = Counter(record.split for record in records)
    by_stroke = Counter(record.stroke_name for record in records)
    by_source = Counter(record.source_class for record in records)
    return {
        "source_root": str(source_root),
        "output_root": str(output_root.resolve()),
        "num_samples": len(records),
        "num_frames": len(records) * args.window_size,
        "splits": dict(sorted(by_split.items())),
        "stroke_counts": dict(sorted(by_stroke.items())),
        "source_class_counts": dict(sorted(by_source.items())),
        "window_size": args.window_size,
        "sampling": args.sampling,
        "pose_model": args.pose_model,
        "pose_conf": args.pose_conf,
        "min_detected_frames": args.min_detected_frames,
        "label_boundary": {
            "stroke_type": "pseudo forehand/backhand inferred from dominant wrist side; VideoBadminton provides action family only",
            "quality_score": "pseudo score; no coach labels were present in source data",
            "pose_labels": "YOLOv8-Pose auto labels; not manually corrected",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert E:/williamsAgentWorkspace/data/pose into this project's data/bsd format."
    )
    parser.add_argument("--source-root", default=r"E:\williamsAgentWorkspace\data\pose")
    parser.add_argument("--output-root", default=str(ROOT / "data" / "bsd"))
    parser.add_argument("--pose-model", default="yolov8n-pose.pt")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu, 0, cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--sampling", choices=["center", "linspace"], default="center")
    parser.add_argument("--min-detected-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-source-class", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only report source counts; do not run pose inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    videos = collect_videos(source_root, args.limit_per_source_class)
    splits = assign_splits(videos, seed=args.seed)

    planned = Counter()
    for video_path, source_class, family in videos:
        planned[(source_class, family, splits[video_path])] += 1
    if args.dry_run:
        print(json.dumps({"num_videos": len(videos), "planned": {str(k): v for k, v in planned.items()}}, indent=2, ensure_ascii=False))
        return

    prepare_output_root(output_root, overwrite=args.overwrite)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to auto-label VideoBadminton clips") from exc

    model = YOLO(args.pose_model)
    records: list[SampleRecord] = []
    rejected: list[dict[str, Any]] = []
    total = len(videos)
    for idx, (video_path, source_class, family) in enumerate(videos, start=1):
        split = splits[video_path]
        try:
            record = process_video(video_path, source_class, family, split, output_root, model, args)
            if record is None:
                rejected.append({"source_video": str(video_path), "reason": "too_few_detected_frames"})
            else:
                records.append(record)
        except Exception as exc:  # keep long conversions resumable by reporting bad clips.
            rejected.append({"source_video": str(video_path), "reason": repr(exc)})
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"[{idx}/{total}] accepted={len(records)} rejected={len(rejected)}", flush=True)

    if not records:
        raise RuntimeError("No usable samples were generated; inspect pose model/device and source videos.")

    assign_template_and_quality(records)
    metadata = make_manifest(output_root, source_root, records, args)
    metadata["rejected_count"] = len(rejected)
    metadata["rejected_examples"] = rejected[:100]

    records_by_split: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        records_by_split[record.split].append(record)
    for split in ("train", "val", "test"):
        write_split_json(output_root, split, records_by_split[split], metadata)
    write_bsd_pose_yaml(output_root)
    (output_root / "conversion_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

