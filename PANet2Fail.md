# 🧪 PANet: Previous Failed Implementations — Detailed Report

## 📌 Project Overview

This report summarizes the implementation strategy, architecture, dataset, experimental details, and failure analysis of two attempts to implement **PANet (Prototype Alignment Network)** from scratch for few-shot medical image segmentation on the **BUS-UCLM** dataset.

---

## 🎯 Objective

To design a **few-shot semantic segmentation framework** using PANet-style prototype alignment to accurately segment tumors in **ultrasound breast images**. The model aimed to perform well even with **limited labeled samples**, simulating few-shot learning conditions (e.g., 1-shot or 5-shot).

---

## 🗂️ Dataset Used

### Dataset: **BUS-UCLM Breast Ultrasound Dataset** (Orville Kaggle Version)

- 📁 Images: 2D grayscale breast ultrasound scans.
- 🏷️ Masks: Pixel-wise tumor annotations (grayscale float masks).
- ⚠️ Problem: **Masks were not binary** — instead ranged from 0 to 0.3 intensity.
- 🧼 Preprocessing:
  - Resized to 256x256.
  - Normalized images between 0–1.
  - ❌ Failed to binarize masks → major segmentation issues.

---

## 🧠 Architecture Implemented

### ✔️ Base Design
- PANet-style dual branch architecture:
  - **Support branch** and **Query branch**.
  - Used shared encoders or shallow CNNs.

### ✔️ Components

| Module              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| 🔹 Encoder          | CNN-based 4-layer encoder with BatchNorm, ReLU, MaxPool.                     |
| 🔹 Prototype Gen    | Averaged masked support features to form the class prototype.                |
| 🔹 Feature Matching | Dot-product or cosine similarity between prototype and query feature.       |
| 🔹 Decoder          | Upsampled the query feature map into final prediction mask.                 |

### ❌ Limitations
- No pretrained backbones (used shallow conv nets).
- Feature vectors and prototypes were not normalized.
- Decoder was too simple (1 conv + upsample).
- No skip connections or multi-scale fusion (unlike PANet original).

---

## ⚙️ Training Setup

| Component           | Value / Description                            |
|---------------------|-------------------------------------------------|
| 🔢 Loss Function    | BCEWithLogitsLoss or DiceLoss                  |
| 📦 Optimizer        | Adam                                            |
| 🔁 Epochs           | 5–10                                            |
| 📏 Batch Size       | 1 (episodic)                                    |
| 💾 Evaluation       | Dice Score                                      |
| 🎯 Few-shot Mode    | 1-shot and 5-shot episodes manually constructed |

---

## 📉 Results (Both Attempts)

| Metric                     | Value     |
|----------------------------|-----------|
| Average Training Loss      | ~1.05–1.30 |
| Validation Dice Score      | ~0.00–0.02 |
| Visual Output              | All black / near-zero predictions |
| Sigmoid Prediction Mean    | ~0.05 (flat) |
| Gradient Flow              | Weak gradients from masked regions |

---

## 🔍 Postmortem: Why the Model Failed

| Issue                             | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| ❌ Mask Binarization Missing     | Masks had soft pixel values (0–0.3) → model couldn’t learn tumor boundaries |
| ❌ Support/Query Construction    | Episodes not correctly randomized or balanced                               |
| ❌ Prototype Calculation Error   | Feature maps not normalized → unstable similarity values                    |
| ❌ Prediction Saturation         | Output map stuck near 0 after sigmoid                                       |
| ❌ Model Depth Insufficient      | Shallow encoder failed to capture tumor features                            |
| ❌ No Visualization or Sanity Check | Lack of mask/image visual feedback during training                         |
| ❌ Loss Function Limitation      | BCE alone didn’t work with fuzzy masks, DiceLoss gradients unstable         |
| ❌ Dataset Issue                 | BUS-UCLM lacks well-structured class-wise annotations for few-shot tasks    |

---

## 🔎 Debug Logs and Observations

- `Pred shape`: torch.Size([1, 1, 256, 256])
- `Pred min/max`: ~0.01–0.17
- `Sigmoid mean`: ~0.05
- **Dice Coefficient**: 0.0000 for multiple epochs
- Training loss plateaued around 1.1–1.2.

---

## 🛠️ Lessons Learned

- Always **threshold masks** during preprocessing.
- Visualize support-query pairs before training.
- Normalize features before cosine similarity.
- Avoid shallow CNNs for medical image encoders.
- Build proper episodic samplers.
- Avoid datasets where masks are poorly annotated or fuzzy.

---

## ✅ Outcome

Both implementations **failed** in producing meaningful segmentation results. However, this helped identify key design flaws and dataset-specific issues. The next version (PANet++) will address these with:

- Better dataset (e.g., SIIM-ACR Pneumothorax)
- Stronger encoder (ResNet/VGG)
- Proper normalization & cosine similarity
- Visual debugging during training
- Better loss design

---

## 📌 Status: FAILED — Ready to Restart with Improvements

