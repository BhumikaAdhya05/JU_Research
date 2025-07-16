# Few-Shot Tumor Segmentation in Breast Ultrasound Images

## 🧪 Project Overview

This repository extends conventional segmentation models (like ResUNet++) to handle **low-data scenarios** using **few-shot learning**, particularly relevant in medical imaging where labeled data is scarce.

We explore three promising few-shot segmentation strategies:

- **PANet (Prototype Alignment Network)**
- **RePRI (Recursive Refinement with Intermediate Supervision)**
- **FSS-1000 Inspired Adaptations**

These methods will be tested on **low-resource medical imaging datasets** like BUS-UCLM, after benchmarking ResUNet++ on BUS-UC.

---

## 📍 Why Few-Shot Segmentation?

Medical datasets often contain very few annotated samples due to:
- Limited availability of expert radiologists.
- Variability in tumor types, shapes, and image quality.

**Few-shot segmentation** allows us to generalize to unseen tumors using only **1–5 labeled examples**, improving the clinical applicability of AI models.

---

## 🔍 Methods Explored

### 1. 🧠 PANet (Prototype Alignment Network)

**Type:** Prototype-based Few-Shot Segmentation  
**Core Idea:**
- Extracts **feature prototypes** from support images (examples).
- Compares query image features to prototypes to predict segmentation.

**Architecture Highlights:**
- Support branch: builds prototypes.
- Query branch: extracts features from new image.
- Alignment module: matches query features to prototypes for pixel-wise classification.

**Pros:**
- Lightweight and fast.
- Works well in both 1-shot and 5-shot settings.
- Ideal for quick deployment and clinical use.

---

### 2. 🧬 RePRI (Recursive Refinement with Intermediate Supervision)

**Type:** Meta-Learning  
**Core Idea:**
- Learns how to adapt its segmentation strategy by refining predictions in steps.
- Uses **intermediate supervision** to improve predictions over iterations.

**Strengths:**
- Handles noise and ambiguity well.
- Strong generalization across unseen tumor types.
- More powerful than pure prototype-based models, but also more complex.

---

### 3. 🧪 FSS-1000 Style Adaptations

**Dataset Origin:** FSS-1000 – A benchmark for few-shot segmentation.  
**Use Case:**
- While FSS-1000 is not medical, its structure (10 samples per class across 1000 classes) makes it ideal for building meta-learning baselines.
- We can **adapt models trained on FSS-1000** (e.g., HSNet, PFENet) to tumor segmentation via transfer learning and fine-tuning.

**Approach:**
- Leverage architectures designed for FSS-1000 and adapt them to BUS-UCLM.
- Evaluate using 1-shot and 5-shot configurations.

---

## 🧭 Strategy Moving Forward

1. ✅ **Baseline Complete**: ResUNet++ implemented and evaluated on the BUS-UC dataset.
2. 🔄 **Next Step**: Switch to a **few-shot compatible dataset** like **BUS-UCLM**.
3. 🧪 **Model Implementation (in order)**:
    - [ ] PANet (Prototype-Based)
    - [ ] RePRI (Meta-Learning)
    - [ ] FSS-1000 Pretrained Model Adaptation
4. 📊 **Evaluation Metrics**:
    - Dice Coefficient (F1 Score)
    - IoU (Jaccard Index)
    - Precision, Recall, Accuracy
    - Few-Shot Generalization Score (mean over N-way K-shot tasks)

---

## 📚 References

- Wang et al., ["PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment"](https://arxiv.org/abs/1908.01998)
- RePRI: ["RePRI: Towards Real-World Few-Shot Semantic Segmentation"](https://arxiv.org/abs/2004.05373)
- FSS-1000 Dataset: ["A New Benchmark Dataset for Few-Shot Segmentation"](https://arxiv.org/abs/2004.05373)
- BUS-UC & BUS-UCLM: Breast Ultrasound Segmentation Datasets from [Kaggle](https://kaggle.com)

---


