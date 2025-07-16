# PANet for Few-Shot Breast Tumor Segmentation (Ultrasound)

This project explores the use of **PANet (Prototype Alignment Network)** for **few-shot medical image segmentation**, applied to breast ultrasound images from the **BUS-UCLM dataset**. It was an experimental attempt to understand how well PANet performs in medical imaging when only a few annotated samples are available for training.

---

## 🧠 What is PANet (Prototype Alignment Network)?

**PANet** in the context of few-shot segmentation refers to **Prototype Alignment Network**, not Pyramid Attention Network.

- Originally proposed in the paper:  
  *"PANet: Few-Shot Image Segmentation with Prototype Alignment"*  
  [https://arxiv.org/abs/2003.10061](https://arxiv.org/abs/2003.10061)

### 🔍 Key Concepts:
- Learns from **just a few support images and masks**
- Generates **class prototypes** from support set features
- Performs **prototype alignment and matching** with query image features
- Trains using **episodic learning** format to mimic few-shot tasks

---

## 🗂️ Dataset

- **Name**: BUS-UCLM (Breast Ultrasound Lesion Segmentation)
- **Source**: Kaggle
- **Used As**:
  - **Support Set**: A few images + masks (1-shot or 5-shot)
  - **Query Set**: Separate test samples for generalization

---

## ⚙️ Implementation Summary

### ✅ Data Preprocessing
- Resized all images and masks to 256×256
- Normalized to [0, 1] range
- Created episodic batches: `(support_images, support_masks, query_images, query_masks)`

### ✅ Model Structure (PANet)

- **Encoder**: ResNet50 (pretrained)
- **Support feature extractor**: Extracts features from few annotated examples
- **Query feature extractor**: Encodes unannotated input
- **Prototype computation**: Computes class prototype from support set
- **Alignment module**: Compares query features with support prototype
- **Decoder**: Outputs binary mask (foreground/background)

### ⚠️ Incomplete Aspects

- ❌ The **alignment module** was not fully implemented
- ❌ No attention refinement in support-query interaction
- ❌ Prototype averaging was fixed, not learned
- ❌ No proper meta-learning loop (fixed support/query batches)
- ❌ Dice or IoU losses were missing; only BCE used

---

## 📉 Observations

- Loss curve shows high variance and poor convergence
- Predictions remained around 0.4–0.5 confidence (weak masks)
- Model failed to sharply distinguish tumors in query images
- Suggests poor generalization from support to query

---

## 🧪 Why the Model Failed

- The PANet architecture for few-shot segmentation **requires precise alignment and prototype fusion**, which was **not implemented** correctly
- Query and support encoding was disconnected
- Model did **not use episodic training correctly**, defeating the few-shot purpose
- The few available training examples made the backbone overfit

---

## 🔁 What Can Be Improved

| Area | Fix |
|------|-----|
| Alignment | Add cosine similarity-based alignment between support/query |
| Prototypes | Use attention-weighted prototypes, not just global mean |
| Decoder | Add spatial-aware decoder (skip connections, UNet-style) |
| Loss | Combine Dice Loss + BCE for better mask learning |
| Episodes | Dynamically sample few-shot episodes per iteration |

---

## ✅ Conclusion

This project attempted to explore **Prototype Alignment Networks (PANet)** for **few-shot breast tumor segmentation** on ultrasound images. While the model did not produce strong segmentation results, the pipeline built here can be upgraded into a correct few-shot learner by:

- Fixing prototype alignment logic
- Implementing proper episodic training
- Adding attention-based refinement modules

This base can serve as a starting point for more robust few-shot segmentation work in medical imaging.

---
