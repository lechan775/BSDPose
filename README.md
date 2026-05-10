# BSDPose: Badminton Stroke Detection via YOLOv8-Pose + CBAM + Kinematic Features

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-FFB81C)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official implementation of the paper **"Badminton Stroke Recognition and Quality Assessment Based on YOLOv8-Pose and Kinematic Feature Analysis"** (submitted to 湖南理工学院学报 自然科学版).

## Highlights

- **CBAM-enhanced YOLOv8-Pose**: Embed CBAM (Channel + Spatial Attention) after PAN P3/P4/P5 output layers to improve keypoint detection under motion blur.
- **15-dim kinematic features → 62-dim aggregation**: Joint angles (shoulder-elbow-wrist, hip-knee-ankle), torso inclination, wrist velocity, CoM displacement. 16-frame window → statistical pooling → 62-D vector.
- **Multi-task head**: Shared encoder (62→128→64) → classification branch (6 stroke types) + regression branch (0–100 quality score). Joint loss: CE + 0.5×MSE.
- **Benchmarked on BSD dataset**: 2743 samples, 43888 frames, 6 stroke classes derived from VideoBadminton.

## Project Structure

```
BSDPose/
├── models/                  # CBAM/SE attention modules + YOLOv8-Pose injection
│   ├── cbam.py              #   ChannelAttention, SpatialAttention, CBAM, SEAttention
│   └── yolov8_cbam_pose.py  #   inject_pan_attention(), build_yolov8_pose_with_attention()
├── features/
│   └── kinematic.py         #   15 per-frame features → 62-D window aggregation
├── classifier/
│   ├── multi_task_head.py   #   MultiTaskHead, FeatureMLPClassifier, SequenceLSTMClassifier
│   └── dataset.py           #   KinematicFeatureDataset, feature record loading
├── train/
│   ├── train_pose.py        #   Train baseline/SE/CBAM YOLOv8-Pose variants
│   └── train_classifier.py  #   Train LSTM/MLP/MultiTask classifiers
├── scripts/
│   └── prepare_bsd.py       #   Build BSD dataset from VideoBadminton .mp4 clips
├── eval/
│   ├── eval_pose.py         #   Evaluate pose models (mAP, PCK)
│   ├── eval_classifier.py   #   Evaluate classifiers (accuracy, MAE)
│   └── eval_ablation.py     #   Run full ablation experiments
├── configs/
│   └── config.yaml          #   All hyperparameters and experiment definitions
└── paper/
    └── journal_draft.md     #   Manuscript (Markdown)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare BSD Dataset

Download [VideoBadminton](https://github.com/your-username/videobadminton) or place your own badminton .mp4 clips under `data/pose/VideoBadminton_Dataset/`, then run:

```bash
python scripts/prepare_bsd.py \
    --source-root data/pose \
    --output-root data/bsd \
    --window-size 16 \
    --seed 42
```

This generates:
- `data/bsd/images/` — extracted 16-frame window images
- `data/bsd/labels/` — YOLO-format pose annotation labels
- `data/bsd/train.json`, `val.json`, `test.json` — classifier feature records
- `data/bsd/bsd-pose.yaml` — Ultralytics dataset config

### 3. Train Pose Estimation Models

```bash
# Train all variants (baseline, SE, CBAM)
python train/train_pose.py --variant all

# Or train specific variant
python train/train_pose.py --variant cbam --epochs 200 --batch 16
```

Checkpoints saved to `logs/pose/`.

### 4. Train Classifiers

```bash
# Train all classifier variants (LSTM, MLP, MultiTask)
python train/train_classifier.py --variant all

# Or specific experiment
python train/train_classifier.py --variant cbam_kinematic_multitask --epochs 200
```

### 5. Evaluate

```bash
# Pose evaluation (mAP on BSD test split)
python eval/eval_pose.py

# Classifier evaluation (accuracy, precision, recall, F1, MAE)
python eval/eval_classifier.py

# Full ablation study
python eval/eval_ablation.py --output-dir results/
```

## Key Results (BSD Test Set)

| Task | Baseline | Ours (CBAM) | Gain |
|------|----------|-------------|------|
| Pose mAP@0.5:0.95 | 82.46% | **83.46%** | +1.00 pp |
| Stroke Accuracy | 56.78% (MLP) | **57.78%** | +1.00 pp |
| Quality MAE | 2.41 | **2.20** | -0.21 |

## Citation

```bibtex
@article{bsdpose2025,
  title    = {基于YOLOv8-Pose与运动学特征分析的羽毛球击球动作识别与质量评估研究},
  author   = {XXX and XXX and 罗正},
  journal  = {湖南理工学院学报(自然科学版)},
  year     = {2026},
  note     = {Submitted}
}
```

## License

MIT License.
