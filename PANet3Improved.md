# 🌟 PANet++: Enhanced Few-Shot Segmentation for Breast Tumor Ultrasound

## 🧠 Motivation

Previous implementations of PANet failed due to issues like poor dataset masks, lack of deep features, and improperly computed prototypes. PANet++ introduces **key architectural and training improvements** to achieve **accurate segmentation from very few annotated examples**.

---

## 🎯 Objective

Implement an **improved prototype alignment network (PANet++)** to perform **1-shot or 5-shot semantic segmentation** on medical ultrasound images, specifically for tumor localization.

---

## 📦 Dataset

### 📌 Primary: **BUSI Breast Ultrasound Dataset** (Kaggle)

- ✅ Classes: Benign, Malignant, Normal
- ✅ 780 Images with corresponding binary masks
- ✅ Clear, sharp tumor annotations
- ✅ Few-shot Simulation: Split into Support and Query dynamically

---

## 🧠 Architecture

### 🔷 Encoder (Backbone)

- `ResNet-50` or `VGG16` pretrained on ImageNet
- Extracts high-level features from support & query images

### 🔷 Prototype Generator

- Support features masked by ground truth mask
- Average pooled to get class prototype vector
- **L2 normalization** ensures cosine distance stability

### 🔷 Feature Matching

- Cosine similarity between query features and prototype
- Can include **temperature scaling (τ)** to control sharpness

### 🔷 Decoder

- PANet++ Decoder:
  - Skip connections from encoder
  - Multi-scale context fusion (dilated convolutions)
  - 3 Conv + Upsample blocks
  - Final 1×1 convolution + sigmoid

---

## ⚙️ Training Pipeline

| Component        | Details                                        |
|------------------|------------------------------------------------|
| Loss Function    | `0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss`     |
| Optimizer        | `Adam` (lr=1e-4)                               |
| Episode Design   | Random 1-shot or 5-shot episodic training loop |
| Batch Size       | 1 (each episode = one support-query set)       |
| Epochs           | 50 (early stopping enabled)                    |
| Evaluation       | Dice Score + IoU                              |

---

## 📈 Expected Results

| Metric             | Value Range (Expected) |
|--------------------|------------------------|
| Dice Coefficient   | ✅ 0.70 – 0.85+         |
| IoU                | ✅ 0.55 – 0.75          |
| Inference Time     | ⚡ Real-time per image  |
| Few-shot Accuracy  | ✅ Learns with ≤ 5 samples |

---

## 🖼️ Visualization Samples

- Support + Mask → Query + Prediction
- Tumor overlays
- Intermediate attention maps
- Live Dice score logging per episode

---

