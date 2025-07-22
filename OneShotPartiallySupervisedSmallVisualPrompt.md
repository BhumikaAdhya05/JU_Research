# 🧬 One-Shot and Partially-Supervised Cell Image Segmentation Using Small Visual Prompt  
**CVPR 2024**

**Authors:** Masahiro Kato (The University of Tokyo), Tomoya Sato (NEC Corporation)  
📄 [Paper Link](https://openaccess.thecvf.com/content/CVPR2024/html/Kato_One-Shot_and_Partially-Supervised_Cell_Image_Segmentation_Using_Small_Visual_CVPR_2024_paper.html)

---

## 🧠 Overview

This paper presents a simple yet powerful method for **segmenting cells in microscopy images**, even when very few (or only one) labeled example is available. Instead of using text prompts or entire support sets like in traditional few-shot learning, it proposes a **Small Visual Prompt** that can generalize across tasks with minimal supervision.

---

## 🎯 Problem Statement

**Cell Segmentation** is key in biomedical imaging. However:

- Annotating data is expensive.
- Class distributions are highly imbalanced.
- Labels for new classes are often unavailable at training time.

🔍 Goal: Perform **one-shot** or **partially supervised** segmentation of unseen or rare cell types using only **1 or few examples**.

---

## 🌟 Key Idea

Use a **Small Visual Prompt** (a 64×64 crop from the support image) instead of:

- Text prompts (as in CLIP)
- Feature prototypes
- Full support sets

This small visual prompt is directly concatenated with the query image and used as context for segmentation.

---

## 🏗️ Method: Small Visual Prompt (SVP)

### 💡 What is SVP?

- A tiny 64×64 crop from a **labeled support image**.
- Contains one or more foreground instances.
- Acts as a query-specific guidance signal.

### 🧩 How It Works

1. **Inputs**:
   - Query image: The image to be segmented.
   - Visual prompt: A small crop from a support image containing target object(s).

2. **Model**: A lightweight CNN architecture that:
   - Processes both prompt and query.
   - Concatenates prompt embedding with query feature.
   - Produces final segmentation map.

3. **Training**:
   - Supervised on base classes (with annotations).
   - Evaluated on novel/unseen classes with just 1 prompt.

4. **Inference**:
   - Just feed a query + a visual prompt.
   - Model segments the matching instances.

---

## 🧪 Experimental Setup

### 🧬 Dataset: LIVECell

- 8 types of cell lines.
- 3 are used as **novel** classes (for evaluation).
- Others are base classes for training.

| Class Type | Used For   |
|------------|------------|
| Base       | Training   |
| Novel      | Inference Only (No train labels) |

### 🧮 Metrics

- **mean IoU (mIoU)**
- **F1 Score**
- **AP (Average Precision)**

---

## 📈 Results

### 🔍 Novel Class Segmentation (1-shot)

| Method           | mIoU (%) | F1 Score | AP (%) |
|------------------|----------|----------|--------|
| CLIP-Adapter     | 13.5     | 31.0     | 16.8   |
| SAM w/ Prompt    | 22.8     | 42.4     | 27.5   |
| Feature Matching | 24.2     | 44.9     | 30.7   |
| Prompt-Only CNN  | 29.1     | 48.6     | 34.8   |
| **Ours (SVP)**   | **42.5** | **60.1** | **49.7** |

✔️ SVP significantly outperforms prompt-based and zero-shot models like CLIP and SAM in one-shot setting.

---

## 🧪 Ablation Studies

| Prompt Type     | mIoU (%) |
|-----------------|-----------|
| No Prompt       | 19.3      |
| Whole Image     | 31.5      |
| 128×128 Patch   | 39.7      |
| **64×64 Patch (Ours)** | **42.5**  |

📌 Small patches are surprisingly powerful!

---

## ⚙️ Model Architecture

- A simple **CNN** model is used.
- Prompt encoder and image encoder share weights.
- Features are concatenated → decoder → segmentation output.
- Trained using standard **Dice + BCE loss**.

---

## 💡 Why Does It Work?

- Cells in the same class often have **local shape/texture similarities**.
- A small patch (prompt) is sufficient to teach the model **what to look for**.
- Unlike text prompts or prototype averaging, this **directly leverages pixel-space context**.

---

## 🧠 Key Takeaways

| Feature                  | Benefit                                    |
|--------------------------|--------------------------------------------|
| Small Visual Prompt      | No need for text, masks, or prototypes     |
| 1-shot Generalization    | Works with just 1 support example          |
| No Pretraining Needed    | Trained from scratch, no large models used |
| Simple & Efficient       | Lightweight CNN architecture               |

---

## 🔍 Comparison to Other Approaches

| Approach          | Needs Text? | Pretraining | Few-shot Friendly | mIoU (1-shot) |
|------------------|-------------|-------------|-------------------|---------------|
| CLIP-Adapter      | ✅          | ✅          | ❌                | 13.5          |
| SAM               | ❌          | ✅          | ❌                | 22.8          |
| Prototypes (PANet)| ❌          | ❌          | ✅                | 24.2          |
| **SVP (Ours)**    | ❌          | ❌          | ✅✅✅              | **42.5**      |

---

## 🧪 One-shot → Multi-shot Extension

The SVP approach can be extended to use **multiple prompts** by:

- Averaging features from multiple patches
- Using attention to select the most relevant ones

But even in pure **1-shot**, it already performs better than other baselines.

---

## 📁 Repository Structure (Expected)

```bash
SVP/
├── models/              # CNN-based segmentation model
├── data/                # LIVECell loader
├── utils/               # Prompt extraction, cropping
├── train.py             # Training script
├── inference.py         # One-shot evaluation
└── README.md            # This file
```
