from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


COCO_KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


FEATURE_NAMES = [
    "right_shoulder_elbow_wrist_angle",
    "left_shoulder_elbow_wrist_angle",
    "right_elbow_shoulder_hip_angle",
    "left_elbow_shoulder_hip_angle",
    "right_hip_knee_ankle_angle",
    "left_hip_knee_ankle_angle",
    "torso_inclination_angle",
    "center_of_mass_x",
    "center_of_mass_y",
    "right_wrist_velocity",
    "left_wrist_velocity",
    "center_of_mass_velocity",
    "right_wrist_acceleration",
    "right_wrist_relative_height",
    "knee_angle_symmetry",
]


@dataclass(frozen=True)
class WindowFeature:
    start: int
    end: int
    features: np.ndarray


def _as_xy_array(keypoints: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 17 or arr.shape[1] < 2:
        raise ValueError("keypoints must have shape (17, 2) or (17, >=3)")
    return arr[:17, :2]


def angle_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle ABC in degrees."""

    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-8:
        return float("nan")
    cos_value = float(np.dot(ba, bc) / denom)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def center_of_mass(keypoints: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    kp = _as_xy_array(keypoints)
    idx = [
        COCO_KEYPOINTS["left_shoulder"],
        COCO_KEYPOINTS["right_shoulder"],
        COCO_KEYPOINTS["left_hip"],
        COCO_KEYPOINTS["right_hip"],
    ]
    return np.nanmean(kp[idx], axis=0)


def torso_inclination_angle(keypoints: np.ndarray | Iterable[Iterable[float]]) -> float:
    kp = _as_xy_array(keypoints)
    shoulder_mid = (kp[COCO_KEYPOINTS["left_shoulder"]] + kp[COCO_KEYPOINTS["right_shoulder"]]) / 2.0
    hip_mid = (kp[COCO_KEYPOINTS["left_hip"]] + kp[COCO_KEYPOINTS["right_hip"]]) / 2.0
    torso_vec = hip_mid - shoulder_mid
    vertical = np.array([0.0, 1.0], dtype=np.float32)
    denom = float(np.linalg.norm(torso_vec) * np.linalg.norm(vertical))
    if denom <= 1e-8:
        return float("nan")
    cos_value = float(np.dot(torso_vec, vertical) / denom)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def extract_frame_features(
    keypoints: np.ndarray | Iterable[Iterable[float]],
    previous_keypoints: np.ndarray | None = None,
    previous_right_wrist_velocity: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Compute the 15 per-frame kinematic features.

    Returns the feature vector and the current right-wrist velocity so the next
    frame can compute acceleration.
    """

    kp = _as_xy_array(keypoints)
    prev = _as_xy_array(previous_keypoints) if previous_keypoints is not None else None
    c = COCO_KEYPOINTS

    right_knee_angle = angle_three_points(kp[c["right_hip"]], kp[c["right_knee"]], kp[c["right_ankle"]])
    left_knee_angle = angle_three_points(kp[c["left_hip"]], kp[c["left_knee"]], kp[c["left_ankle"]])
    com = center_of_mass(kp)

    if prev is None:
        right_wrist_velocity = 0.0
        left_wrist_velocity = 0.0
        com_velocity = 0.0
    else:
        prev_com = center_of_mass(prev)
        right_wrist_velocity = float(np.linalg.norm(kp[c["right_wrist"]] - prev[c["right_wrist"]]))
        left_wrist_velocity = float(np.linalg.norm(kp[c["left_wrist"]] - prev[c["left_wrist"]]))
        com_velocity = float(np.linalg.norm(com - prev_com))

    right_wrist_acceleration = right_wrist_velocity - previous_right_wrist_velocity
    features = np.array(
        [
            angle_three_points(kp[c["right_shoulder"]], kp[c["right_elbow"]], kp[c["right_wrist"]]),
            angle_three_points(kp[c["left_shoulder"]], kp[c["left_elbow"]], kp[c["left_wrist"]]),
            angle_three_points(kp[c["right_elbow"]], kp[c["right_shoulder"]], kp[c["right_hip"]]),
            angle_three_points(kp[c["left_elbow"]], kp[c["left_shoulder"]], kp[c["left_hip"]]),
            right_knee_angle,
            left_knee_angle,
            torso_inclination_angle(kp),
            float(com[0]),
            float(com[1]),
            right_wrist_velocity,
            left_wrist_velocity,
            com_velocity,
            right_wrist_acceleration,
            float(kp[c["nose"], 1] - kp[c["right_wrist"], 1]),
            abs(left_knee_angle - right_knee_angle),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), right_wrist_velocity


def extract_sequence_features(keypoint_sequence: np.ndarray | Iterable[Iterable[Iterable[float]]]) -> np.ndarray:
    """Compute a (T, 15) feature matrix from a COCO-17 keypoint sequence."""

    sequence = np.asarray(keypoint_sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1] < 17 or sequence.shape[2] < 2:
        raise ValueError("keypoint_sequence must have shape (T, 17, 2) or (T, 17, >=3)")

    rows: list[np.ndarray] = []
    previous = None
    previous_right_wrist_velocity = 0.0
    for frame in sequence:
        features, previous_right_wrist_velocity = extract_frame_features(
            frame,
            previous_keypoints=previous,
            previous_right_wrist_velocity=previous_right_wrist_velocity,
        )
        rows.append(features)
        previous = frame
    return np.vstack(rows).astype(np.float32)


def aggregate_window_features(
    keypoint_window: np.ndarray | Iterable[Iterable[Iterable[float]]],
) -> np.ndarray:
    """Aggregate a 16-frame keypoint window into the required 62-D vector."""

    sequence = np.asarray(keypoint_window, dtype=np.float32)
    frame_features = extract_sequence_features(sequence)
    stats = [
        np.nanmean(frame_features, axis=0),
        np.nanstd(frame_features, axis=0),
        np.nanmax(frame_features, axis=0),
        np.nanmin(frame_features, axis=0),
    ]
    center_start = center_of_mass(sequence[0])
    center_end = center_of_mass(sequence[-1])
    displacement = center_end - center_start
    magnitude = float(np.linalg.norm(displacement))
    direction = float(math.atan2(float(displacement[1]), float(displacement[0])))
    output = np.concatenate([*stats, np.array([magnitude, direction], dtype=np.float32)])
    if output.shape != (62,):
        raise AssertionError(f"Expected 62 features, got {output.shape}")
    return np.nan_to_num(output.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def sliding_window_features(
    keypoint_sequence: np.ndarray | Iterable[Iterable[Iterable[float]]],
    window_size: int = 16,
    stride: int = 8,
) -> list[WindowFeature]:
    """Compute 62-D features for sliding windows over a keypoint sequence."""

    sequence = np.asarray(keypoint_sequence, dtype=np.float32)
    if sequence.ndim != 3:
        raise ValueError("keypoint_sequence must be a 3-D array")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if len(sequence) < window_size:
        return []

    windows: list[WindowFeature] = []
    for start in range(0, len(sequence) - window_size + 1, stride):
        end = start + window_size
        windows.append(WindowFeature(start=start, end=end, features=aggregate_window_features(sequence[start:end])))
    return windows

