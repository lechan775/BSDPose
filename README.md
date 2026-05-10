<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Ultralytics-YOLOv8-FFB81C?logo=yolo" alt="YOLOv8">
<img src="https://img.shields.io/badge/License-MIT-green" alt="License">
<img src="https://img.shields.io/badge/Paper-湖南理工学报-red" alt="Paper">
<br>
<img src="https://img.shields.io/badge/Pose%20mAP-83.46%25-blue" alt="Pose mAP">
<img src="https://img.shields.io/badge/Stroke%20Acc-57.78%25-green" alt="Accuracy">
<img src="https://img.shields.io/badge/Quality%20MAE-2.20-orange" alt="MAE">

</div>

<br>

**BSDPose** is an open-source framework for badminton stroke recognition and quality assessment, integrating **YOLOv8-Pose** with **CBAM attention** and **kinematic feature analysis**. It supports six-class stroke classification and 0–100 quality scoring in a single multi-task pipeline.

> 📄 This repository accompanies the paper *"Badminton Stroke Recognition and Quality Assessment Based on YOLOv8-Pose and Kinematic Feature Analysis"* (湖南理工学院学报·自然科学版, 2026).

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **CBAM-Enhanced YOLOv8-Pose** | Channel + Spatial attention injected at PAN P3/P4/P5 outputs for better keypoint localization under motion blur |
| 🏸 **15-D Kinematic Features** | Joint angles, torso inclination, wrist velocity, CoM displacement → statistical pooling over 16-frame windows → 62-D vector |
| 🧠 **Multi-Task Head** | Shared encoder (62→128→64) + classification branch (6 strokes) + regression branch (0–100 quality), joint loss: CE + 0.5·MSE |
| 📊 **Full Experiment Pipeline** | End-to-end: data preparation → pose training (baseline/SE/CBAM) → feature extraction → classifier training → evaluation |

## 📦 Installation

<details open>
<summary>Pip install (recommended)</summary>

```bash
git clone https://github.com/lechan775/BSDPose.git
cd BSDPose
pip install -r requirements.txt
```

Requirements: `torch>=2.0`, `ultralytics>=8.0`, `opencv-python`, `pyyaml`, `numpy`

</details>

## 🏃 Quick Start

<details open>
<summary>1. Prepare BSD Dataset</summary>

```bash
# Place VideoBadminton .mp4 clips under data/pose/VideoBadminton_Dataset/
python scripts/prepare_bsd.py \
    --source-root data/pose \
    --output-root data/bsd \
    --window-size 16 \
    --seed 42
```

Output: `data/bsd/{images,labels,train.json,val.json,test.json,bsd-pose.yaml}`

</details>

<details open>
<summary>2. Train Pose Estimation</summary>

```bash
# All three variants (baseline, SE, CBAM)
python train/train_pose.py --variant all --epochs 200 --batch 16

# Single variant
python train/train_pose.py --variant cbam --device 0
```

</details>

<details open>
<summary>3. Train Classifier</summary>

```bash
# All classifier experiments
python train/train_classifier.py --variant all --epochs 200 --batch-size 64

# Our proposed method only
python train/train_classifier.py --variant cbam_kinematic_multitask
```

</details>

<details open>
<summary>4. Evaluate</summary>

```bash
python eval/eval_pose.py        # Pose mAP on BSD test split
python eval/eval_classifier.py  # Accuracy / Precision / Recall / F1 / MAE
python eval/eval_ablation.py    # Full ablation results
```

</details>

## 📊 Benchmark Results

### Pose Estimation (BSD Test Split)

| Model | mAP@0.5:0.95 (%) |
|-------|-----------------:|
| YOLOv8-Pose (*baseline*) | 82.46 |
| + SE | 82.19 |
| + CBAM (**ours**) | **83.46** |

### Stroke Classification (BSD Test Split)

| Method | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|--------|------------:|--------------:|-----------:|-------------:|
| OpenPose + LSTM | 45.79 | 44.00 | 45.79 | 43.49 |
| YOLOv8-Pose + LSTM | 46.15 | 44.00 | 46.15 | 43.29 |
| YOLOv8-Pose + Kinematic + MLP | 56.78 | 57.80 | 56.78 | 57.11 |
| **YOLOv8-Pose + CBAM + Kinematic + Multi-Task (ours)** | **57.78** | **58.80** | **57.78** | **58.11** |

### Ablation Study

| Pose Model | CBAM | Kinematic | Multi-Task | Accuracy (%) | Quality MAE |
|------------|:----:|:---------:|:----------:|-------------:|------------:|
| YOLOv8-Pose | ✗ | ✗ | ✗ | 46.15 | — |
| YOLOv8-Pose | ✓ | ✗ | ✗ | 44.50 | — |
| YOLOv8-Pose | ✗ | ✓ | ✓ | 53.48 | 2.41 |
| YOLOv8-Pose | ✓ | ✓ | ✓ | **57.78** | **2.20** |

### Key Findings

- **CBAM alone does not improve** classification when used with raw LSTM (44.50% vs 46.15%)
- **CBAM + kinematic features synergy**: combined use boosts accuracy by +4.30 pp (53.48% → 57.78%)
- **Quality MAE reduction**: multi-task head compresses MAE from 2.41 to 2.20

## 🧩 Project Structure

```
BSDPose/
├── models/                      # Attention modules & YOLOv8-Pose injection
│   ├── cbam.py                  #   ChannelAttention, SpatialAttention, CBAM, SEAttention
│   └── yolov8_cbam_pose.py      #   inject_pan_attention(), build_yolov8_pose_with_attention()
├── features/
│   └── kinematic.py             #   15 per-frame features → 62-D window aggregation
├── classifier/
│   ├── multi_task_head.py       #   MultiTaskHead, MultiTaskLoss, MLP/LSTM baselines
│   └── dataset.py               #   KinematicFeatureDataset, FeatureRecord loader
├── train/
│   ├── train_pose.py            #   Train baseline/SE/CBAM YOLOv8-Pose variants
│   └── train_classifier.py      #   Train LSTM/MLP/MultiTask classifiers
├── scripts/
│   └── prepare_bsd.py           #   VideoBadminton → BSD dataset builder
├── eval/
│   ├── eval_pose.py             #   Pose evaluation (mAP)
│   ├── eval_classifier.py       #   Classifier evaluation (accuracy, MAE)
│   └── eval_ablation.py         #   Ablation experiment runner
├── configs/
│   └── config.yaml              #   Hyperparameters & experiment matrix
└── paper/
    └── journal_draft.md         #   Manuscript (Markdown)
```

## 🔬 Methodology

```
VideoBadminton (.mp4)
        │
        ▼
  16-Frame Window Sampling
        │
        ▼
┌───────────────────────────────┐
│  YOLOv8-Pose Variants         │
│  ┌─────────┬────────┬───────┐ │
│  │Baseline │  + SE  │ +CBAM │ │
│  │82.46%   │ 82.19% │83.46% │ │
│  └─────────┴────────┴───────┘ │
└───────────────────────────────┘
        │
        ▼  COCO-17 Keypoints (x,y,conf)
┌───────────────────────────────────┐
│  Kinematic Feature Extraction      │
│  15 per-frame → 16-frame window    │
│  → 62-D vector (mean/std/max/min)  │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  Multi-Task Head                   │
│  Shared:  62 → 128 → 64           │
│  ┌──────────────┬────────────────┐ │
│  │ Classifier   │   Regressor    │ │
│  │ 64→32→6      │   64→32→1      │ │
│  │ (Softmax)    │   (Sigmoid)    │ │
│  └──────────────┴────────────────┘ │
│  L = CE + 0.5·MSE                  │
└───────────────────────────────────┘
        │
        ▼
  Output: 6-Class Stroke + 0–100 Quality Score
```

## 🏷️ Stroke Classes

| # | English | 中文 |
|---|---------|------|
| 0 | Forehand Clear | 正手高远球 |
| 1 | Backhand Clear | 反手高远球 |
| 2 | Forehand Smash | 正手杀球 |
| 3 | Backhand Smash | 反手杀球 |
| 4 | Forehand Net Lift | 正手网前挑球 |
| 5 | Backhand Net Lift | 反手网前挑球 |

## 📖 Citation

```bibtex
@article{bsdpose2026,
  title     = {{基于YOLOv8-Pose与运动学特征分析的羽毛球击球动作识别与质量评估研究}},
  author    = {胡竞文 and 罗正 and 潘润杭 and 范飞豪},
  journal   = {湖南理工学院学报(自然科学版)},
  year      = {2026},
  note      = {Submitted}
}
```

## 📝 License

This project is licensed under the [MIT License](LICENSE) — free for academic and commercial use.

---

<div align="center">
  <sub>Built with ❤️ for Badminton AI Research</sub>
</div>
