from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from features.kinematic import aggregate_window_features, extract_sequence_features


STROKE_NAMES = [
    "forehand_clear",
    "backhand_clear",
    "forehand_smash",
    "backhand_smash",
    "forehand_net_lift",
    "backhand_net_lift",
]


@dataclass
class FeatureRecord:
    features: np.ndarray
    label: int
    quality_score: float
    sequence_features: np.ndarray | None = None
    meta: dict[str, Any] | None = None


def _reshape_keypoints(flat_or_nested: Any) -> np.ndarray:
    arr = np.asarray(flat_or_nested, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        elif arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[0] < 17 or arr.shape[1] < 2:
        raise ValueError("keypoints must describe at least 17 points with x/y coordinates")
    return arr[:17, : min(arr.shape[1], 3)]


def _record_from_sample(sample: dict[str, Any]) -> FeatureRecord:
    label = int(sample["stroke_type"])
    quality = float(sample.get("quality_score", 0.0))
    meta = {k: v for k, v in sample.items() if k not in {"features", "keypoints_sequence", "frame_features"}}

    if "features" in sample:
        features = np.asarray(sample["features"], dtype=np.float32)
        if features.shape != (62,):
            raise ValueError(f"features must have shape (62,), got {features.shape}")
        if "frame_features" in sample:
            sequence_features = np.asarray(sample["frame_features"], dtype=np.float32)
        elif "keypoints_sequence" in sample:
            sequence_features = extract_sequence_features(np.asarray(sample["keypoints_sequence"], dtype=np.float32))
        else:
            sequence_features = None
        return FeatureRecord(features=features, label=label, quality_score=quality, sequence_features=sequence_features, meta=meta)

    if "keypoints_sequence" in sample:
        keypoints_sequence = np.asarray(sample["keypoints_sequence"], dtype=np.float32)
        features = aggregate_window_features(keypoints_sequence)
        sequence_features = extract_sequence_features(keypoints_sequence)
        return FeatureRecord(features=features, label=label, quality_score=quality, sequence_features=sequence_features, meta=meta)

    raise ValueError("sample must contain either 'features' or 'keypoints_sequence'")


def _records_from_coco_annotations(data: dict[str, Any], window_size: int = 16, stride: int = 8) -> list[FeatureRecord]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        if "keypoints_sequence" in ann or "features" in ann:
            return [_record_from_sample(item) for item in data.get("annotations", [])]
        key = str(ann.get("segment_id", ann.get("video_id", ann.get("clip_id", ann.get("image_id")))))
        grouped.setdefault(key, []).append(ann)

    records: list[FeatureRecord] = []
    for group_key, annotations in grouped.items():
        annotations.sort(key=lambda x: int(x.get("frame_index", x.get("image_id", 0))))
        if not annotations:
            continue
        if "stroke_type" not in annotations[0]:
            continue
        keypoints = [_reshape_keypoints(ann["keypoints"]) for ann in annotations if "keypoints" in ann]
        if len(keypoints) < window_size:
            continue
        sequence = np.stack(keypoints, axis=0)
        for start in range(0, len(sequence) - window_size + 1, stride):
            end = start + window_size
            window = sequence[start:end]
            records.append(
                FeatureRecord(
                    features=aggregate_window_features(window),
                    label=int(annotations[start].get("stroke_type", annotations[0]["stroke_type"])),
                    quality_score=float(annotations[start].get("quality_score", annotations[0].get("quality_score", 0.0))),
                    sequence_features=extract_sequence_features(window),
                    meta={"segment_id": group_key, "start": start, "end": end},
                )
            )
    return records


def load_feature_records(path: str | Path, window_size: int = 16, stride: int = 8) -> list[FeatureRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = [_record_from_sample(sample) for sample in data]
    elif "samples" in data:
        records = [_record_from_sample(sample) for sample in data["samples"]]
    elif "annotations" in data:
        records = _records_from_coco_annotations(data, window_size=window_size, stride=stride)
    else:
        raise ValueError("Dataset JSON must contain a list, 'samples', or COCO-style 'annotations'")

    if not records:
        raise ValueError(f"No feature records could be built from {path}")
    return records


class KinematicFeatureDataset(Dataset):
    """Dataset returning 62-D features or 16x15 sequence features plus labels."""

    def __init__(
        self,
        json_path: str | Path,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        return_sequence: bool = False,
        window_size: int = 16,
        stride: int = 8,
    ) -> None:
        self.records = load_feature_records(json_path, window_size=window_size, stride=stride)
        self.return_sequence = return_sequence
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        if self.return_sequence and any(record.sequence_features is None for record in self.records):
            raise ValueError("return_sequence=True requires frame_features or keypoints_sequence in every record")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        if self.return_sequence:
            assert record.sequence_features is not None
            x = record.sequence_features.astype(np.float32)
        else:
            x = record.features.astype(np.float32)
            if self.feature_mean is not None and self.feature_std is not None:
                x = (x - self.feature_mean) / np.maximum(self.feature_std, 1e-6)

        return {
            "features": torch.from_numpy(x.astype(np.float32)),
            "label": torch.tensor(record.label, dtype=torch.long),
            "quality_score": torch.tensor(record.quality_score, dtype=torch.float32),
        }


def compute_feature_normalizer(records: list[FeatureRecord]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([record.features for record in records], axis=0).astype(np.float32)
    return matrix.mean(axis=0), matrix.std(axis=0)
