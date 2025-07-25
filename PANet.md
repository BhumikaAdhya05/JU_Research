# 🧠 PANet for Few-Shot Tumor Segmentation in Breast Ultrasound Images

This repository contains an implementation of **PANet (Prototype Alignment Network)** for **few-shot semantic segmentation** on the **BUS-UCLM breast ultrasound dataset**, focusing on tumor region segmentation using limited annotated examples.

> Implemented under the guidance of **Professor Ram Sarkar**, Department of Computer Science, Jadavpur University, as part of a research internship.

---

## 📌 Objective

In real-world medical imaging scenarios, annotated data is scarce. This project explores **PANet** to perform **few-shot tumor segmentation** where only 1–5 labeled images are available for training.

---

## 🧩 Dataset

- **BUS-UCLM** *(Breast Ultrasound)*  
  Source: Kaggle  
  - Grayscale ultrasound images  
  - Annotated binary tumor masks  
  - Ideal for simulating 1-shot / 5-shot segmentation tasks

**Few-Shot Setup:**
- Episodes are created using:
  - `N-shot` support images (with masks)
  - 1 query image (predict tumor mask)

---

---

## 🔧 Model Design Strategy: Implementing PANet from Scratch

This project implements **PANet (Prototype Alignment Network) from scratch**, tailored for **grayscale breast ultrasound tumor segmentation** using few-shot learning.

While we use pretrained backbones (e.g., ResNet-50 or VGG) for feature extraction, the **core PANet logic is entirely custom-built** to support 1-shot and 5-shot semantic segmentation.

### 🔨 Components Built from Scratch:

| Component                | Description                                             | Built From Scratch? |
|-------------------------|---------------------------------------------------------|----------------------|
| **Data Loader**          | Episodic support-query sampling for few-shot learning   | ✅ Yes               |
| **Preprocessing**        | Normalization, resizing, grayscale handling             | ✅ Yes               |
| **Prototype Computation**| Averages support features to form class representations | ✅ Yes               |
| **Similarity Module**    | Cosine similarity or inner product matching             | ✅ Yes               |
| **Segmentation Head**    | Decoder for similarity map → segmentation mask          | ✅ Yes               |
| **Training Loop**        | Episodic training with few-shot tasks                   | ✅ Yes               |
| **Evaluation Metrics**   | Dice, IoU, Accuracy, Precision, Recall                  | ✅ Yes               |
| **Feature Encoder**      | CNN (e.g., ResNet-50 or ViT, optionally pretrained)     | 🟡 Partially         |

This design ensures **maximum flexibility**, supports custom datasets like BUS-UCLM, and makes the system **fully transparent for research and reproducibility**.

---

## 🧠 Methodology

### ✅ Model: PANet (Prototype Alignment Network)

PANet performs segmentation by computing **class prototypes** from support masks and **aligning** them with the query image features.

### 🔍 Key Components

| Module              | Description |
|---------------------|-------------|
| **Shared Encoder**  | Extracts features from both support and query images (e.g., ResNet-50 or ViT). |
| **Prototype Module**| Averages foreground features from support masks to form class prototypes. |
| **Similarity Matching** | Computes pixel-wise similarity between query features and prototypes. |
| **Segmentation Head** | Converts similarity map into predicted segmentation mask. |

---

## 🏗️ Implementation Plan

### 📁 1. Dataset Preparation
- Normalize grayscale images.
- Resize to uniform size (e.g., 256×256).
- Create episodes (support + query splits).
- Binarize masks (0 = background, 1 = tumor).

### 🧠 2. Model Construction
- Encoder (e.g., ResNet/VGG).
- Prototype extractor (average over foreground pixels).
- Similarity module (cosine similarity or inner product).
- Decoder head to map similarity to pixel mask.

### 🧪 3. Training Setup
- **Episodic training**: randomly sample episodes per iteration.
- **Loss functions**:
  - Binary Cross-Entropy
  - Dice Loss
- **Evaluation metrics**:
  - Dice Score (F1)
  - IoU (Jaccard)
  - Accuracy, Precision, Recall

### 📊 4. Few-Shot Evaluation
- 1-shot and 5-shot settings
- Evaluate on held-out query images using sampled support sets
- Report per-episode performance

---

## 🔧 Requirements

- Python 3.8+
- PyTorch / TensorFlow (to be confirmed)
- OpenCV, NumPy, scikit-image, matplotlib

Install with:

```bash
pip install -r requirements.txt
```
