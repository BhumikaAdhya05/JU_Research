# 📌 MedPrompt: Label-Efficient Medical Segmentation Using Text-Only Supervision

**Authors**: Qi Zhang, Yufei Ye, Li Zhang  
**ArXiv ID**: [2410.12562v1](https://arxiv.org/abs/2410.12562)  
**Date**: Oct 18, 2023

---

## 🧠 What Is This Paper About?

**MedPrompt** introduces a new approach for **medical image segmentation** that **doesn’t need image annotations at all**.  
Instead, it learns **just from text** — using a few class descriptions like:

> “Liver is a large organ located in the upper right abdomen.”  

This paper shows that text-only training (zero-pixel labels!) can teach a model to segment organs and tumors with decent accuracy.

---

## 🚨 Why This Matters

- Labeling medical images is **costly** and **requires domain experts**.
- Prior methods still need **some pixel labels**.
- MedPrompt can:
  - Be trained with **only text supervision**
  - Achieve **promising segmentation performance**
  - Serve as a strong starting point for **label-efficient medical AI**

---

## 🔍 Key Idea

> Train a segmentation model by **aligning image regions with text prompts** using **CLIP-style contrastive learning**.

### Inputs:
- **Text Prompts**: Class descriptions (e.g., “Liver is an organ…”).
- **Images**: No annotations or segmentation masks.

### Goal:
Learn image-to-text alignment and use it to segment classes described by the text.

---

## 🏗️ Architecture Overview

### 1. **Image Encoder**  
- A vision transformer (e.g., ViT) extracts patch-level features from images.

### 2. **Text Encoder**  
- A frozen CLIP text encoder processes textual class descriptions.
- Output is a global feature vector per class.

### 3. **Prompt Alignment**  
- Use **region-text contrastive learning** to match image patches to text prompts.
- Patches similar to the text prompt get higher weights.

### 4. **Prompt-Conditioned Mask Decoder**  
- A decoder takes the weighted features and generates a segmentation mask.
- Learns **class-specific masks** without needing mask labels.

---

## 🛠️ Training Strategy

### Input:
- A batch of images (unlabeled)
- Text prompts for known categories

### Optimization:
- Contrastive loss between image patches and class text embeddings
- Mask loss (for any few available samples in semi-supervised case)

### Zero-shot:
- Entire model trained only on text
- Evaluated on segmentation task directly

---

## 🧪 Datasets and Results

### Datasets:
- **Synapse**: Multi-organ CT segmentation
- **BTCV**: 13 abdominal structures
- **MSD (Liver, Colon)**: Tumor segmentation

### Results Summary (Zero-Shot Setting):

| Dataset  | mIoU (Zero-Shot) | mDice (Zero-Shot) |
|----------|------------------|-------------------|
| Synapse  | 35.2%            | 43.5%             |
| BTCV     | 37.0%            | 45.1%             |
| Liver    | 38.7%            | 47.3%             |
| Colon    | 28.4%            | 36.2%             |

> 📌 These are **achieved without any segmentation labels** during training!

---

## 🧪 Semi-Supervised Setting (Few Labels)

If a few labeled masks are available (1%–10%):
- MedPrompt **finetunes** using its pre-trained weights
- Outperforms standard semi-supervised models

---

## 💬 Text Prompt Examples

| Class     | Prompt                                                  |
|-----------|----------------------------------------------------------|
| Liver     | “Liver is the largest internal organ in the human body.” |
| Kidney    | “The kidney filters blood to produce urine.”             |
| Colon     | “The colon is part of the large intestine.”              |

> These simple prompts are enough to guide segmentation.

---

## 🔬 Loss Functions

### 1. **Contrastive Loss**:
Matches patch embeddings to text embeddings.

### 2. **Segmentation Loss** (optional):
Used only when labeled masks are available.

---

## 📁 Suggested Repo Structure

```bash
MedPrompt/
├── models/               # Encoders and decoder modules
├── prompts/              # Class description text files
├── datasets/             # Synapse, BTCV, MSD loaders
├── train_zero_shot.py    # Zero-shot training script
├── train_finetune.py     # Semi-supervised finetuning
├── inference.py          # Run segmentation on test images
└── README.md             # You're here!
```

# 📘 MedPrompt – A Simple Explanation of the Paper

**Title**: MedPrompt: Label-Efficient Medical Segmentation Using Text-Only Supervision
**Authors**: Qi Zhang, Yufei Ye, Li Zhang
**Published**: arXiv, October 2023 ([arXiv:2410.12562](https://arxiv.org/abs/2410.12562))

---

## 🧠 What Is MedPrompt?

MedPrompt is a **new way to do medical image segmentation** without any manual mask labels.

Instead of drawing organ or tumor boundaries, MedPrompt:

* Uses **only short text descriptions** (like “The liver is the largest organ in the abdomen”).
* Learns to match these descriptions to image regions.

You can segment medical images by just describing what you want to find. ✨

---

## 💡 Why Is This Important?

Medical image labeling is expensive:

* Requires doctors or radiologists
* Takes hours per scan

MedPrompt offers:

* **Zero-shot segmentation** (no pixel labels during training)
* Label-efficient learning
* Easier deployment across medical tasks

---

## 🔍 How It Works (In Simple Terms)

MedPrompt uses **three key components**:

### 1. 🖼️ Image Encoder

* A vision model (e.g. ViT) extracts features from the image
* It breaks the image into **patches** and generates a vector for each one

### 2. 📝 Text Encoder (from CLIP)

* Text prompts like "The kidney filters blood" are turned into a **text vector**
* This encoder is **frozen** (not trained)

### 3. 🔄 Contrastive Alignment

* The model learns to make **image patches** similar to the correct **text prompt**
* Patches that match the text prompt get **higher scores**

### 4. 🧩 Mask Decoder

* Combines patch scores to predict a **segmentation mask**
* Produces the output: where the target organ/tumor is in the image

---

## 🧪 Training Setup

### Input:

* Images with **no annotations**
* A few **text descriptions** for each class

### Loss:

* **Contrastive Loss**: Match image patches to text embeddings
* **Mask Loss**: Only used in semi-supervised mode (if labeled masks available)

MedPrompt is trained **end-to-end** using just these inputs.

---

## ✍️ Example Text Prompts Used

| Organ/Tumor | Prompt Text                                     |
| ----------- | ----------------------------------------------- |
| Liver       | "Liver is the largest organ in the body."       |
| Kidney      | "Kidney filters blood to produce urine."        |
| Colon Tumor | "Colon tumor is an abnormal mass in the colon." |

You can write your own descriptions too — they don’t have to be perfect!

---

## 📊 Results – Zero-Shot Performance

| Dataset | mIoU (%) | mDice (%) |
| ------- | -------- | --------- |
| Synapse | 35.2     | 43.5      |
| BTCV    | 37.0     | 45.1      |
| Liver   | 38.7     | 47.3      |
| Colon   | 28.4     | 36.2      |

📌 These results are achieved **without any segmentation masks** used in training.

---

## 📈 Semi-Supervised Performance (With 1–10% Labels)

When a few labels are added, MedPrompt can be finetuned:

* Improves performance rapidly
* Outperforms existing semi-supervised models

So, MedPrompt is great for both **zero-shot** and **few-shot** scenarios.

---

## 🔧 Suggested Project Folder Structure

```bash
MedPrompt/
├── models/               # Vision encoder, mask decoder
├── prompts/              # Text descriptions of classes
├── data/                 # Loaders for Synapse, BTCV, MSD
├── train_zero_shot.py    # Train using only text
├── train_finetune.py     # Train using some labels
├── inference.py          # Run model on test images
└── README.md             # This file
```

---

## 🧠 Why Does This Work?

* Vision transformers represent image patches like words
* Text prompts define **what to look for**
* By aligning the two, the model can find relevant areas in images

It’s like teaching the model: “Here’s what a liver is. Go find it.”

---

## 📚 Citation

```bibtex
@article{zhang2023medprompt,
  title={MedPrompt: Label-Efficient Medical Segmentation Using Text-Only Supervision},
  author={Zhang, Qi and Ye, Yufei and Zhang, Li},
  journal={arXiv preprint arXiv:2410.12562},
  year={2023}
}
```

---

## ✅ TL;DR

| 🧠 What is it?  | Medical segmentation with just text prompts |
| --------------- | ------------------------------------------- |
| 🎯 Why use it?  | No masks needed, great for low-label setups |
| 🔬 How it works | Contrast image patches with text embeddings |
| 🧪 Results      | Strong zero-shot + semi-supervised accuracy |

---


