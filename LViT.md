# 🧠 LViT: Language Meets Vision Transformer in Medical Image Segmentation

> **Published in:** IEEE Transactions on Medical Imaging, Vol. 43, No. 1, Jan 2024  
> **Authors:** Zihan Li, Yunxiang Li, Qingde Li, et al.  
> **DOI:** [10.1109/TMI.2023.3291719](https://doi.org/10.1109/TMI.2023.3291719)  
> **Official Code:** [github.com/HUANGLIZI/LViT](https://github.com/HUANGLIZI/LViT)

---

## 📝 Abstract

Medical image segmentation suffers from the lack of high-quality labeled data due to annotation costs. **LViT** (Language meets Vision Transformer) is a **semi-supervised** segmentation model that fuses **medical text annotations** with image features using a **hybrid CNN-Transformer** architecture. Key innovations include:

- **Text-guided segmentation**
- **Pixel-Level Attention Module (PLAM)**
- **Exponential Pseudo-label Iteration (EPI)**
- **Language-Vision (LV) Loss**

---

## 🔬 Motivation

Challenges in medical image segmentation:
1. **Limited annotated data** due to expensive expert labeling.
2. **Poor image quality** or unclear region boundaries.

💡 **Solution**: Use *medical text annotations* (e.g., radiology reports) to guide image segmentation, leveraging **semi-supervised learning** to generate high-quality pseudo-labels.

---

## 📌 Key Contributions

| Innovation | Description |
|-----------|-------------|
| **LViT Model** | Double-U architecture combining CNN and Vision Transformer |
| **Text Embedding** | Direct embedding (not full text encoder) for lightweight text integration |
| **PLAM** | Enhances local feature extraction in CNN through channel & spatial attention |
| **EPI Mechanism** | Uses Exponential Moving Average to refine pseudo labels in semi-supervised setting |
| **LV Loss** | Supervision for unlabeled images using cosine similarity between image & text masks |

---

## 🧠 Methodology

### 🔹 U-Shape CNN Branch
- Acts as segmentation head.
- Uses Conv → BN → ReLU blocks.
- Downsamples via MaxPool.
- Merges features with ViT outputs and PLAM.

### 🔹 U-Shape ViT Branch
- Processes combined **image + text embeddings**.
- Uses BERT (12 layers) to embed structured text.
- PatchEmbedding + Multi-Head Self-Attention (MHSA).
- Performs **cross-modal fusion**.

### 🔹 PLAM (Pixel-Level Attention Module)
- Combines GAP and GMP in parallel branches.
- Helps retain **local features** lost by ViT.
- Reduces mis-segmentation.

### 🔹 EPI (Exponential Pseudo-Label Iteration)
- Pseudo labels are updated as:

$$
L_{LV} = 1 - \cos(x_{\text{img}, p}, x_{\text{img}, c})
$$

(where β = 0.99, using EMA)
- Prevents deterioration from noisy pseudo-labels.

### 🔹 LV (Language-Vision) Loss
- Uses cosine similarity between:
- Text embeddings
- Corresponding pseudo-labels
- Objective:

$$
P_t = \beta \cdot P_{t-1} + (1 - \beta) \cdot P_t
$$

---

## 📊 Datasets

| Dataset         | Modality | Text Annotations | Description |
|----------------|----------|------------------|-------------|
| **MosMedData+** | CT       | ✅ Structured     | COVID-19 lung infection |
| **QaTa-COV19**  | X-ray    | ✅ Structured     | COVID-19 X-rays with infection regions |
| **ESO-CT**      | CT       | ✅ Rough location | Esophageal cancer CT scans |

---

## 📈 Results

### ✅ Performance (Dice Score / mIoU)

| Model          | MosMedData+ | QaTa-COV19 | ESO-CT  |
|----------------|-------------|------------|---------|
| **LViT-T**     | 74.57 / 61.33 | 83.66 / 75.11 | 71.53 / 59.94 |
| nnUNet         | 72.42 / 60.18 | 80.42 / 70.81 | 70.07 / 57.29 |
| UNet++         | 70.35 / 58.12 | 78.10 / 68.00 | 68.12 / 55.61 |
| TransUNet      | 73.01 / 60.22 | 80.45 / 71.12 | 69.80 / 57.12 |

> ✨ Even with **1/4 labeled data**, LViT matches or exceeds full supervision performance!

---

## ⚗️ Ablation Studies

- **Text Embedding vs. Text Encoder**: Embedding layer uses fewer parameters with comparable or better results.
- **EPI + LV Loss**: Significantly improve pseudo-label stability.
- **Model Variants (Tiny/Small/Base)**:
- More layers (ViT-B) reduce standard deviation but not always improve accuracy.
- **PLAM Components**: Best performance with both GAP and GMP combined.

---

## 🔍 Interpretability

### GradCAM Highlights:

- LViT accurately localizes lesion areas.
- Text prompts improve attention activation and reduce false positives.
- Layer-wise activation shows that **text information sharpens early layer focus** (e.g., DownViT1).

---

## ⚙️ Training Details

- **Framework:** PyTorch
- **Hardware:** 2×Tesla V100, 128GB RAM
- **Loss Function:**  
- Supervised: Dice + CE  
- Unsupervised: Dice + CE + α·LV Loss (α = 0.1)
- **Evaluation Metrics:** Dice Score, mIoU
- **Optimizer:** Adam
- **Learning Rate:**  
- 3e-4 (QaTa-COV19)  
- 1e-3 (MosMedData+)
- **Batch Size:** 24 (default)

---

## 🏁 Future Work

- **3D LViT**: Extension for volumetric medical data.
- **Automatic Text Annotation**: Convert text generation into a classification task.
- **Text-Free Inference**: Enable LViT to operate without text input at test time.

---

## 📚 Citation

```bibtex
@article{li2024lvitt,
title={LViT: Language Meets Vision Transformer in Medical Image Segmentation},
author={Li, Zihan and Li, Yunxiang and Li, Qingde and Wang, Puyang and Guo, Dazhou and Lu, Le and Jin, Dakai and Zhang, You and Hong, Qingqi},
journal={IEEE Transactions on Medical Imaging},
volume={43},
number={1},
pages={96--107},
year={2024},
publisher={IEEE}
}
```
# 🧠 What’s This Paper About?

Medical image segmentation means drawing outlines around organs or diseased regions (like lung infections) in medical scans like CT or X-rays.

This is usually done with deep learning models (like U-Net), but those models need a lot of labeled images—and labeling medical images is:

- 🕒 Time-consuming  
- 🧠 Requires expert doctors  
- 💸 Very expensive

---

## 💡 The Core Idea

The authors noticed that:

> Medical images often come with written text reports (e.g., "infection in lower right lung").

These free-text annotations are:

- Already available (no extra cost)
- Created by doctors
- Helpful in identifying **where** the problem is

So... why not use the **text + image together** to improve segmentation?

---

## 🚀 Introducing: LViT

**LViT** stands for **Language meets Vision Transformer**.

It’s a new AI model that:

- Takes **both medical images and their text reports** as input
- Combines them using a **hybrid architecture (CNN + Transformer)**
- Works well even with **few labeled images**, thanks to smart **pseudo-labels**

---

## 🧱 Architecture in Simple Terms

Think of LViT like a two-path highway:

- 🛣️ **Image Path** – Processes image using CNN (focuses on local details)
- 🛣️ **Multimodal Path** – Processes image + text using Transformer (captures global context)

They **meet at checkpoints**, exchange information, and cooperate to produce better segmentation.

---

## 🧩 Key Components (Explained Simply)

### 1. 🔡 Text Embedding Layer

- Instead of using a heavy language model, it uses a **lightweight embedding layer**
- Turns text into numbers
- Fast & resource-efficient

### 2. 🎯 PLAM (Pixel-Level Attention Module)

- Helps the CNN **focus on small, fuzzy regions** (like early infections)
- Uses:
  - **GAP** (Global Average Pooling)
  - **GMP** (Global Max Pooling)
- Combines spatial & channel-wise info

### 3. 🔁 EPI (Exponential Pseudo-label Iteration)

- Trains on **unlabeled data** using **smart guesses**
- Refines pseudo-labels **over time** with **Exponential Moving Average**
- Prevents bad pseudo-labels from derailing training

🧠 *Analogy*: Blends multiple earlier guesses into a smart average instead of trusting one blindly

### 4. 🧠 LV Loss (Language-Vision Loss)

- Uses **text reports** to guide training
- Checks if predicted mask aligns with text using **cosine similarity**
- If mismatch → high loss → model learns to fix it

---

## 📊 Datasets Used

| Dataset       | Type | Description                              |
|---------------|------|------------------------------------------|
| MosMedData+   | CT   | COVID-19 lung infection (CT)             |
| QaTa-COV19    | X-ray| COVID-19 X-rays + expert text            |
| ESO-CT        | CT   | Esophageal cancer + location annotations |

Each includes **segmentation masks** + **text reports**

---

## ✅ Results in a Nutshell

### Fully Supervised:

LViT **outperforms** U-Net, TransUNet, nnUNet.

### Semi-Supervised (25% labeled data):

Still **beats most fully supervised models** 🤯

### Generalization:

- On new datasets like **ESO-CT**, LViT **still excels**
- Shows model is **robust and adaptable**

---

## 🔍 Interpretability (Why It Works)

Used **GradCAM** to visualize model focus:

- 🧠 Traditional models: Sometimes looked at wrong areas
- 💡 LViT: Focused on the right spots, thanks to **text guidance**

---

## 🛠️ Technical Setup

- **Framework**: PyTorch
- **Optimizer**: Adam
- **Loss**:
  - Supervised → Dice + CrossEntropy
  - Unsupervised → Dice + CE + LV Loss
- **Training Tricks**:
  - Early stopping
  - Smart learning schedules

---

## 🧪 Ablation Studies

| Component         | Benefit                                 |
|------------------|------------------------------------------|
| Text Input        | Big boost in segmentation accuracy       |
| PLAM              | Retains important local details          |
| EPI               | Improves pseudo-label quality            |
| LV Loss           | Helps train on unlabeled data            |
| Embedding Layer   | Faster & lighter than full text encoder  |

---

## 🔮 Future Plans

- Extend from **2D → 3D segmentation**
- **Auto-generate text** from images → No manual reports needed
- Make **text input optional** during inference

---

## 🎯 In One Sentence

**LViT is a smart AI model that combines medical images and doctor-written text to boost segmentation accuracy—especially when labeled data is scarce.**


