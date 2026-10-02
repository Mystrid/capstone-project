# EGASN — Epipolar Graph-Attention Segmentation Network

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **EGASN** is a hybrid segmentation framework for autonomous vehicle road scene understanding combining a MobileNetV2+ASPP backbone, OpenCV epipolar graph construction, cross-attention temporal fusion, and evidential deep learning for uncertainty estimation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
  - [1. Spatial Feature Extraction Backbone](#1-spatial-feature-extraction-backbone)
  - [2. Epipolar Graph Construction via OpenCV](#2-epipolar-graph-construction-via-opencv)
  - [3. Cross-Attention Temporal Gating](#3-cross-attention-temporal-gating)
  - [4. Evidential Deep Learning for Uncertainty](#4-evidential-deep-learning-for-uncertainty)
  - [Architectural Flow Diagram](#architectural-flow-diagram)
- [Benchmark Results](#benchmark-results)
  - [Main Quantitative Results](#main-quantitative-results)
  - [Ablation Study](#ablation-study)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Video Inference Demo](#video-inference-demo)
  - [Edge FPS Benchmarking](#edge-fps-benchmarking)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

Real-world autonomous navigation demands semantic scene understanding that simultaneously satisfies **high spatial fidelity**, **temporal continuity**, and **real-time edge latency**. Contemporary single-frame deep convolutional networks achieve strong per-frame accuracy (mIoU) but suffer from catastrophic inter-frame flickering. Conversely, traditional video semantic segmentation (VSS) networks rely on heavy deep optical flow models (e.g., FlowNet, RAFT) that violate edge thermal and compute budgets (sub-10 FPS).

**EGASN** bridges this divide through a hybrid architecture:

1. Replaces compute-heavy deep optical flow with **real-time classical visual odometry (OpenCV ORB + epipolar geometry)**.
2. Formulates a **spatio-temporal graph** where edges are validated by the fundamental matrix $p_t^T F p_{t-1} = 0$.
3. Employs a **Cross-Attention Gating module** that selectively propagates semantic features along confident geometric tracks while filtering occlusions.
4. Integrates **Evidential Deep Learning (Dirichlet distribution)** to provide calibrated spatial uncertainty maps for fail-safe vehicle decision-making.

---

## Key Features

- **Real-Time Edge Throughput:** Runs at **38.6 FPS** on NVIDIA Jetson AGX Orin (vs. 4.5 FPS for deep-flow warping baselines).
- **Temporal Stability:** Achieves **81.2% fwIoU** (flow-warped IoU), virtually eliminating high-frequency pedestrian and pole flickering.
- **Robust Spatial Accuracy:** Attains **77.4% mIoU** on the 19 Cityscapes evaluation classes, within 1% of heavy ResNet-50 models.
- **Calibrated Uncertainty Estimation:** Outputs pixel-level vacuity uncertainty $U = K / S$ via Dirichlet hyper-parameters, enabling fail-safe maneuvers.
- **Low Compute Footprint:** Uses an edge-optimized MobileNetV2 backbone and pruned ASPP with dilations `[6, 12, 18]`.

---

## Architecture Overview

```
                        ┌──────────────────────────────┐
                        │   Video Stream (I_{t-1}, I_t) │
                        └──────────────┬───────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │      Thread 1: CNN        │                 │    Thread 2: Classical    │
  │    MobileNetV2 + ASPP     │                 │   OpenCV Visual Odometry  │
  └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                │ Spatial Features X_t                        │ ORB + FLANN Match
                │                                             ▼
                │                               ┌───────────────────────────┐
                │                               │     Epipolar RANSAC       │
                │                               │    p_t^T F p_{t-1} = 0    │
                │                               └─────────────┬─────────────┘
                │                                             │ Inlier Graph Edges
                │                                             ▼
                │                               ┌───────────────────────────┐
                │                               │   Delaunay Interpolation  │
                │                               │      Pseudo-Dense Flow    │
                │                               └─────────────┬─────────────┘
                │                                             │ Flow Field
                │      ┌────────────────────────┐             │
                │      │ Feature Warping W(·)   │◄────────────┘
                │      │ X_{t-1} -> X_t^warped  │
                │      └───────────┬────────────┘
                │                  │ Warped Features
                ▼                  ▼
  ┌───────────────────────────────────────────────────────────┐
  │                Cross-Attention Gating                     │
  │   Q = W_q X_t  |  K = W_k X_t^warped  |  V = W_v X_t^warped│
  │            A = Softmax(Q K^T / sqrt(d))                   │
  │              X_fused = X_t + λ (A · V)                    │
  └─────────────────────────────┬─────────────────────────────┘
                                │ Fused Spatio-Temporal Map
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │            Decoder & Evidential Output Head               │
  │      Predictions P_t (19 Classes)  |  Uncertainty U_t     │
  └───────────────────────────────────────────────────────────┘
```

### 1. Spatial Feature Extraction Backbone

- **Base Encoder:** MobileNetV2 utilizing depthwise-separable inverted residual bottlenecks, delivering an optimal balance of representational capacity and parameter efficiency.
- **Pruned ASPP:** An Atrous Spatial Pyramid Pooling module with dilations $[6, 12, 18]$, extracting contextual information across multiple receptive fields into a 256-channel intermediate representation $X_t \in \mathbb{R}^{256 \times H' \times W'}$.

### 2. Epipolar Graph Construction via OpenCV

In parallel to the CNN forward pass:

- **ORB Feature Extraction:** Detects up to 2000 multi-scale keypoints and FAST binary descriptors on grayscale frames $I_{t-1}$ and $I_t$.
- **FLANN Matching & Ratio Test:** Multi-probe LSH index matching filtered via Lowe's ratio test ($\tau = 0.75$).
- **Epipolar Geometric Verification:** Evaluates the Fundamental Matrix $F \in \mathbb{R}^{3 \times 3}$ using RANSAC:
  $$p_t^T F p_{t-1} = 0 \quad \forall (p_{t-1}, p_t) \in \mathcal{M}$$
  Matches that exceed the pixel-distance epipolar threshold ($3.0\,\text{px}$) are discarded as dynamic outliers or texture artifacts.
- **Delaunay Triangulation:** Projects surviving inliers onto a Delaunay mesh over the image plane to interpolate sparse motion vectors into a smooth, pseudo-dense flow field $\mathcal{F}_{t-1 \to t}$.

### 3. Cross-Attention Temporal Gating

Rather than naive feature averaging (which corrupts spatial representations when tracking fails during occlusion), EGASN employs transformer cross-attention:

- **Projections:**
  $$Q = W_q X_t, \quad K = W_k X_t^{warped}, \quad V = W_v X_t^{warped}$$
- **Reliability Attention Map:**
  $$A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$
- **Residual Fusion:**
  $$X_{fused} = X_t + \lambda (A \cdot V)$$
  where $\lambda$ is a learnable scaling factor (initialized to $0.1$). When visual appearance deviates due to occlusion or track loss, attention weights naturally decay to zero, reverting smoothly to pure spatial CNN representations.

### 4. Evidential Deep Learning for Uncertainty

EGASN parameterizes a Dirichlet distribution $\text{Dir}(\boldsymbol{\alpha})$ over class probabilities:

- $\alpha_k = e_k + 1$, where $e_k \ge 0$ represents evidence from the Dirichlet head.
- Expected Class Probability: $\hat{p}_k = \alpha_k / S$, where $S = \sum_{c=1}^K \alpha_c$.
- Spatial Uncertainty: $U = K / S \in (0, 1]$. High uncertainty identifies camera occlusion, sensor dirt, or out-of-distribution environments.
- Objective function balances Expected MSE, a Kullback-Leibler (KL) divergence regularizer with annealing factor $\beta$, and a Temporal Consistency penalty $\mathcal{L}_{TC}$:
  $$\mathcal{L} = \sum_{i=1}^{N} \left( \mathcal{L}_{mse}(P_{t,i}, Y_i) + \beta \cdot \text{KL}[D(\alpha_i) \parallel D(\mathbf{1})] \right) + \gamma_{tc} \mathcal{L}_{TC}$$

---

## Benchmark Results

Evaluated on the **Cityscapes Video Dataset** (19 semantic classes, 500 validation video sequences, recorded at 17 FPS). Edge latency is measured on an **NVIDIA Jetson AGX Orin** platform (64GB unified memory, PyTorch 2.0 + OpenCV 4.8).

### Main Quantitative Results

| Architecture                         | Core Paradigm                  | mIoU (%) ↑ | fwIoU (%) ↑ | FPS (Edge Orin) ↑ |
| :----------------------------------- | :----------------------------- | :--------: | :---------: | :---------------: |
| **SegNet** _(Badrinarayanan et al.)_ | Pure Spatial (Light)           |    60.5    |    52.1     |     **65.0**      |
| **DeepLabV3+** _(Chen et al.)_       | Pure Spatial (Heavy ResNet-50) |  **78.3**  |    64.5     |       18.2        |
| **NetWarp** _(Gadde et al.)_         | CNN Deep Optical Flow Fusion   |    80.1    |    71.3     |        4.5        |
| **EGASN (Ours)**                     | **Graph-Attention + OpenCV**   |  **77.4**  |  **81.2**   |     **38.6**      |

> **Takeaway:** EGASN achieves an unprecedented **81.2% fwIoU**, practically eliminating inter-frame flickering while maintaining **38.6 FPS**—making it uniquely viable for autonomous vehicle closed-loop control.

### Ablation Study

| Spatial CNN |   OpenCV Graph    | Cross-Attention Gate | mIoU (%) | fwIoU (%) |   FPS    |
| :---------: | :---------------: | :------------------: | :------: | :-------: | :------: |
|      ✓      |                   |                      |   74.2   |   58.4    |   45.1   |
|      ✓      | ✓ (Naive Average) |                      |   72.8   |   72.1    |   41.3   |
|      ✓      |         ✓         |    ✓ (Full EGASN)    | **77.4** | **81.2**  | **38.6** |

---

## Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.8 (recommended for GPU acceleration)
- CMake and C++ build tools (for OpenCV compilation if building from source)

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone https://github.com/capstone-project/egasn.git
cd egasn/framework

# 2. Create virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install EGASN package in editable mode
pip install -e .
```

---

## Dataset Preparation

Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/):

- `leftImg8bit_sequence_trainvaltest.zip` (video frame sequences)
- `gtFine_trainvaltest.zip` (fine pixel-level annotations)

Extract into your dataset directory:

```text
cityscapes/
├── leftImg8bit_sequence/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

Update `configs/egasn_cityscapes.yaml`:

```yaml
data:
  root: /path/to/cityscapes
```

---

## Usage

### Training

Train EGASN on Cityscapes video sequences with cosine learning rate scheduling and evidential beta annealing:

```bash
python scripts/train.py --config configs/egasn_cityscapes.yaml --device cuda
```

To resume from a checkpoint:

```bash
python scripts/train.py --config configs/egasn_cityscapes.yaml --resume checkpoints/epoch_50.pth
```

### Evaluation

Evaluate mean IoU (mIoU) and temporal consistency on the validation set:

```bash
python scripts/evaluate.py \
    --config configs/egasn_cityscapes.yaml \
    --checkpoint checkpoints/best_model.pth \
    --device cuda
```

### Video Inference Demo

Generate a color-coded semantic segmentation video with temporal stabilization:

```bash
python scripts/demo_video.py \
    --video assets/input_dashcam.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --config configs/egasn_cityscapes.yaml \
    --output demo_output.mp4 \
    --device cuda
```

### Edge FPS Benchmarking

Benchmark latency breakdown across the OpenCV graph, CNN backbone, cross-attention, and decoder:

```bash
python scripts/benchmark_fps.py \
    --config configs/egasn_cityscapes.yaml \
    --num_iterations 100 \
    --device cuda
```

---

## Configuration

All model, graph, training, and data parameters are defined in `configs/egasn_cityscapes.yaml`:

```yaml
model:
  backbone: mobilenet_v2
  aspp_dilations: [6, 12, 18]
  feature_channels: 256
  num_classes: 19
  attention_heads: 4
  lambda_init: 0.1

graph:
  orb_features: 2000
  flann_table_number: 6
  flann_key_size: 12
  flann_multi_probe_level: 1
  ransac_threshold: 3.0
  ratio_test: 0.75
  min_inliers: 8

training:
  epochs: 200
  batch_size: 4
  lr: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  crop_size: [768, 768]
  beta_annealing_epochs: 10
  gamma_tc: 0.5

data:
  root: /path/to/cityscapes
  num_workers: 4
  input_size: [1024, 2048]
```

---

## Repository Structure

```text
framework/
├── configs/
│   └── egasn_cityscapes.yaml      # Cityscapes training & model configuration
├── egasn/
│   ├── __init__.py                # Package version (0.1.0)
│   ├── data/
│   │   ├── __init__.py            # Exports CityscapesVideoDataset
│   │   ├── cityscapes.py          # Cityscapes loader interface
│   │   ├── cityscapes_video.py    # Temporal pair dataset implementation
│   │   └── transforms.py          # Joint spatial/temporal augmentations
│   ├── graph/
│   │   ├── __init__.py            # Exports EpipolarGraphConstructor
│   │   ├── epipolar.py            # Graph constructor and warper interface
│   │   ├── epipolar_filter.py     # Fundamental matrix RANSAC filtering
│   │   ├── feature_warper.py      # Bilinear feature grid warping
│   │   ├── flow_interpolator.py   # Delaunay triangulation interpolation
│   │   ├── keypoint_extractor.py  # OpenCV ORB & FLANN matcher
│   │   └── pipeline.py            # End-to-end graph construction pipeline
│   ├── losses/
│   │   ├── __init__.py            # Exports EvidentialLoss, TemporalConsistencyLoss
│   │   ├── evidential.py          # Dirichlet EDL loss & annealing
│   │   ├── evidential_loss.py     # Core evidential calculation implementation
│   │   └── temporal_consistency.py# Flow-warped temporal consistency penalty
│   ├── models/
│   │   ├── __init__.py            # Exports EGASN architecture
│   │   ├── backbone.py            # MobileNetV2 encoder + Pruned ASPP
│   │   ├── cross_attention.py     # Cross-attention gating module
│   │   ├── decoder.py             # Feature decoder & Dirichlet head
│   │   └── egasn.py               # Top-level EGASN model definition
│   └── utils/
│       ├── __init__.py            # Exports compute_miou, compute_fwiou
│       ├── metrics.py             # Evaluation metrics (mIoU, fwIoU)
│       ├── uncertainty.py         # Dirichlet vacuity uncertainty calculation
│       └── visualization.py       # Cityscapes color palette & visualizer
├── scripts/
│   ├── benchmark_fps.py           # Latency and FPS benchmarking suite
│   ├── demo_video.py              # Video stream inference & rendering
│   ├── evaluate.py                # Validation evaluation script
│   └── train.py                   # Distributed/single-GPU training script
├── tests/
│   ├── test_backbone.py           # CNN backbone unit tests
│   ├── test_cross_attention.py    # Cross-attention gating unit tests
│   ├── test_epipolar_graph.py     # OpenCV graph construction tests
│   └── test_evidential_loss.py    # Dirichlet loss & annealing tests
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies specification
└── setup.py                       # Setuptools packaging configuration
```

---
