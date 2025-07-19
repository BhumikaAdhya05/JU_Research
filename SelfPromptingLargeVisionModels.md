# 🧠 Self-prompting Large Vision Models for Few-Shot Medical Image Segmentation

> **Authors:** Qi Wu, Yuyao Zhang, Marawan Elbatel  
> **Conference:** DART 2023 (LNCS 14293)  
> [📜 Official Paper DOI](https://doi.org/10.1007/978-3-031-45857-6_16)  
> [💻 Code](https://github.com/PeterYYZhang/few-shot-self-prompt-SAM)

---

## 📌 Motivation

- Annotating medical images is expensive and time-consuming.
- The goal is to achieve **few-shot medical image segmentation** by leveraging the **promptable nature of the Segment Anything Model (SAM)**.
- The core idea: **self-generate prompts** using a **lightweight logistic regression classifier** over SAM’s image embeddings.

---

## 🧠 Key Contributions

- A **training-free (or very light)** self-prompting unit is proposed to generate SAM-compatible prompts using only a few labeled samples.
- Demonstrated **superior performance** over SAM fine-tuning baselines like MedSAM and SAMed under few-shot constraints.
- Can be trained in **< 30 seconds on GPU** or seconds on CPU, making it practical in clinical settings.

---

## 🏗️ Architecture Overview

1. Input image is passed through **frozen SAM encoder** → generates 64×64×256 feature embeddings.
2. These embeddings are input to a **logistic regression** classifier (1×1) → outputs **coarse segmentation mask**.
3. Using this mask:
   - A **bounding box** is generated via morphological ops.
   - A **location point** is extracted using **distance transform**.
4. These **prompts (box + point)** are then passed into SAM decoder → **final segmentation** output.

---

## 🧮 Training Objective

Each pixel is classified using **logistic regression**.

Let:

- `t_{mn}`: ground truth pixel value at position (m, n)
- `ŷ_{mn}`: sigmoid output of the logistic regression at (m, n)

The **pixel-wise cross-entropy loss** is:

$$
L = -\frac{1}{k} \sum_q \sum_{m,n} \left[ t_{mn} \log(\hat{y}_{mn}) + (1 - t_{mn}) \log(1 - \hat{y}_{mn}) \right]
$$

- $t_{mn}$: Ground truth pixel value at position $(m, n)$  
- $\hat{y}_{mn}$: Sigmoid output of the logistic regression at $(m, n)$

Where:
- `k`: number of training images
- `q`: index of image in training set

---

## 🧪 Experiments

### 🗂️ Datasets

- **Kvasir-SEG**: Polyp segmentation, 1000 images
- **ISIC-2018**: Skin lesion segmentation, 2594 images

### ⚙️ Implementation

- **Backbone**: SAM ViT-B (smallest version)
- **Classifier**: Logistic Regression (scikit-learn, default params)
- **Training**: 5-fold cross-validation
- **Post-processing**: 
  - 3× Erosion → 5× Dilation using 5×5 kernel
  - Resize output mask to 256×256
- **Prompt Generation**: Morphology + distance transform

---

## 📊 Results

### 🎯 Quantitative (20-shot)

| Model               | Kvasir Dice | ISIC Dice |
|--------------------|-------------|------------|
| **Ours (Point+Box)** | **62.78%**  | **66.78%** |
| MedSAM [16]         | 55.01%      | 64.94%     |
| SAMed [29]          | 61.48%      | 63.27%     |
| SAM-B (unprompted)  | 52.66%      | 45.25%     |
| Prompted SAM-B (GT) | 86.86%      | 84.28%     |
| U-Net (full-data)   | 88.10%      | 88.36%     |

---

## 🧪 Ablation Studies

### 📌 Number of Shots (ISIC-2018)

| Shots | Linear | Point | Box | Point+Box |
|-------|--------|-------|-----|-----------|
| 10    | 58.81  | 59.49 | 47.13 | 64.22     |
| 20    | 62.69  | 61.23 | 47.99 | 66.78     |
| 40    | 63.81  | 61.52 | 49.81 | 67.88     |
| Full  | 65.62  | 61.76 | 55.03 | 69.51     |

---

## ⚖️ Prompt Type Impact

- **Point** → Helps accurately locate center of object, lacks size/shape info.
- **Box** → Conveys rough region and size but can miss fine localization.
- **Combined** → Achieves the best performance by combining both precision and scope.

---

## ⚠️ Limitations

- ❌ **Multi-instance segmentation**: Struggles to handle multiple objects in an image.
- ❌ **Modality-specific generalization**: SAM decoder underperforms on **ultrasound** due to sensitivity to high-frequency noise.
- ❌ **Decoder bottleneck**: SAM's decoder isn't trained on medical modalities, limiting full potential.

---

## 💡 Future Directions

- ✅ Incorporate **modality-specific fine-tuning** (e.g., MedSAM decoder adapters).
- ✅ Use **iterative prompt refinement** for better multi-stage segmentation.
- ✅ Replace logistic regression with **lightweight CNNs or attention-based modules**.
- ✅ Explore **multi-instance prompt generation** techniques (clustering, region growing).

---

## 📎 References

- MedSAM: [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)
- SAMed: [arXiv:2304.13785](https://arxiv.org/abs/2304.13785)
- SAM (Segment Anything): [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

---

## 🔗 Useful Links

- 📄 [Springer DOI](https://doi.org/10.1007/978-3-031-45857-6_16)
- 💻 [GitHub Repository](https://github.com/PeterYYZhang/few-shot-self-prompt-SAM)

---

## 🧾 Citation

```bibtex
@inproceedings{wu2023selfprompting,
  title={Self-prompting Large Vision Models for Few-Shot Medical Image Segmentation},
  author={Wu, Qi and Zhang, Yuyao and Elbatel, Marawan},
  booktitle={DART 2023},
  year={2024},
  publisher={Springer}
}
```

# 🧠 Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation

> A simplified and detailed explanation of the paper by Qi Wu, Yuyao Zhang, and Marawan Elbatel  
> Presented at DART 2023 ([Paper Link](https://doi.org/10.1007/978-3-031-45857-6_16))

---

## 🚨 The Problem

Medical image segmentation — like finding tumors in MRI or polyps in endoscopy — needs:
- ✅ Highly accurate models
- ❌ Lots of **manual annotations** by medical experts
- ❌ Expensive labeling effort

But what if we only have **a few labeled images** (few-shot)? How do we still make a model that segments well?

---

## 🧪 The Solution: Self-Prompting SAM

### What's SAM?

- SAM = **Segment Anything Model** from Meta AI
- It’s trained on huge datasets of **natural images**
- It can segment objects **if you give it prompts**, like:
  - A **point** inside the object
  - A **bounding box** around the object

But here’s the issue:
> SAM needs **good prompts** to work well. And in medical imaging, it’s hard to provide these manually for each new scan.

### So what do the authors propose?

> ✨ Train a tiny model to **generate those prompts** automatically — using just a few labeled images — and feed them to SAM.

---

## 🛠️ Step-by-Step: How It Works

### 🔒 1. Freeze SAM

- Don’t touch SAM’s internal parameters.
- Use it **as-is**: image encoder + prompt encoder + mask decoder.

---

### 🧠 2. Add a Tiny Self-Prompting Module

- Input an image into SAM’s ViT encoder → get **64×64×256** feature embeddings
- Add a **simple logistic regression classifier** (just a linear layer!)
  - It predicts: for each pixel, is it inside the object? (Yes = 1, No = 0)
- This gives a **rough mask** — not perfect, but enough to hint where the object is.

---

### 🧾 3. Extract Prompts from the Rough Mask

From the coarse binary mask:
- 🟡 **Point Prompt**:
  - Use **distance transform** to find the pixel farthest from the boundary (i.e. safely inside the object)
- 🔲 **Box Prompt**:
  - Find the smallest box that covers the mask
  - Clean the mask with **morphological operations** (erosion + dilation)

---

### 🎯 4. Use Prompts with SAM

- Combine the **point** and **box** with the original image
- Feed these into SAM's **mask decoder**
- 🧠 SAM uses its pre-trained power to generate a **precise segmentation mask**

---

## 🤖 What Makes This Special?

### ✅ Works with Very Little Data

- Only needs **10–20 labeled images** to perform well
- No need to train or fine-tune the big SAM model

### ✅ Ultra-Lightweight

- The classifier is just a **single linear layer**
- Whole training takes **<30 seconds on a GPU**, or a few seconds on CPU!

### ✅ Beats Other Fine-Tuning Methods in Few-Shot Settings

- Performs better than MedSAM and SAMed when both use the same small dataset
- Without touching SAM’s weights at all

---

## 📊 Performance Summary (20-shot setting)

| Model               | Kvasir Dice | ISIC Dice |
|--------------------|-------------|------------|
| Ours (point + box) | **62.78%**  | **66.78%** |
| MedSAM             | 55.01%      | 64.94%     |
| SAMed              | 61.48%      | 63.27%     |
| SAM (no prompts)   | 52.66%      | 45.25%     |
| Full-data U-Net    | 88.10%      | 88.36%     |

---

## 🔬 Why This Works (Intuitively)

> Think of the model as two brains working together:

### 🧠 Brain 1: "Rough Estimator" (the self-prompt module)

- Learns **basic shape and location** of the object from just a few images
- Not perfect, but cheap and fast

### 🧠 Brain 2: "Expert Segmenter" (SAM)

- Given a decent prompt (box + point), it uses its **deep knowledge from millions of natural images** to do the real work

**Result:** High-quality segmentation using very little training data.

---

## ⚖️ Limitations

- ❌ Doesn’t handle **multiple objects** in one image well
- ❌ Struggles with **noisy modalities** like ultrasound
- ❌ SAM's decoder isn't trained for medical data — can still be a bottleneck

---

## 💡 Future Improvements

- Combine with **modality-tuned decoders** for better accuracy
- Replace linear classifier with a **lightweight CNN**
- Improve **multi-instance prompting**
- Explore **iterative refinement**: better prompts → better masks → better prompts...

---

## 🧠 Summary: What You Should Remember

- You can use a **frozen giant model** (SAM) and teach a **tiny helper** to give it rough instructions
- This works surprisingly well — even in **few-shot settings**
- Self-prompting turns **coarse predictions** into **high-quality segmentations**
- It’s **simple**, **fast**, and **effective** for real-world medical scenarios

---


