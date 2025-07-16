# PANet-based Segmentation on Breast Ultrasound Images

This project is an experimental implementation of the **Pyramid Attention Network (PANet)** applied to the **BUS-UCLM breast ultrasound dataset** for tumor segmentation. The main objective was to explore how well a simplified PANet-style architecture can perform in low-data medical image segmentation tasks.

---

## 🧠 What is PANet?

**PANet** (Pyramid Attention Network) was originally designed for instance segmentation in natural images. It builds on top of FPN (Feature Pyramid Networks) and enhances both bottom-up and top-down pathways with:

- **Adaptive Feature Pooling**
- **Bottom-up Path Augmentation**
- **Attention modules for better localization**
- **Mask refinement branches**

In this project, a **partial and simplified version** of PANet was implemented to test feasibility on medical data.

---

## 📦 Dataset Used

- **Name**: BUS-UCLM (Breast Ultrasound Lesion Segmentation Dataset)
- **Source**: Kaggle
- **Contents**:
  - Breast ultrasound images in `.png` format
  - Corresponding binary segmentation masks

The dataset was manually extracted, and paths were organized into two folders:
- `images/` for input ultrasound scans
- `masks/` for corresponding binary masks

---

## ⚙️ What Was Implemented

This project implemented a **simplified PANet-like architecture**:

### ✅ Data Pipeline

- Custom PyTorch `Dataset` class to load and preprocess images
- Preprocessing:
  - Resizing to 256x256
  - Normalization to [0, 1]
  - Optional tensor formatting for model input

### ✅ Model Architecture

- Backbone: **ResNet-50** (pretrained on ImageNet)
- Only the **top-down feature extraction** path from PANet was loosely retained
- Final segmentation head: single 1×1 convolution followed by bilinear upsampling and sigmoid activation
- No actual attention modules, pooling, or refinement branches were implemented

### ⚠️ Limitations in PANet Implementation

- ❌ No Bottom-up path augmentation
- ❌ No Adaptive Feature Pooling
- ❌ No explicit Attention mechanisms
- ❌ No multi-scale feature fusion
- ❌ No skip connections or lateral fusion from lower layers
- ❌ No multi-branch mask refinement

---

## 🧪 Training Setup

- Loss: **Binary Cross-Entropy**
- Optimizer: **Adam**
- Epochs: 200
- Device: GPU (Colab T4)
- Metrics like Dice or IoU were not used for validation

---

## 📉 Observations and Failure Points

- Training loss decreased very slowly and plateaued early
- Final segmentation masks were **blurry and under-confident**
- Model failed to **capture sharp tumor boundaries**
- Partial PANet structure failed to leverage spatial hierarchies effectively
- The network **over-relied on high-level ResNet features** and lacked multi-scale context

---

## 🔧 What Could Be Improved

1. ✅ Implement full PANet as proposed in the original paper:
   - Add bottom-up path augmentation
   - Integrate adaptive feature pooling
   - Introduce spatial & channel-wise attention mechanisms
   - Include mask refinement stages

2. ✅ Add Dice and IoU metrics to guide learning

3. ✅ Improve image augmentation and regularization

4. ✅ Add skip connections for better spatial detail retention

5. ✅ Try different loss functions (e.g., Dice, Focal)

6. ✅ Consider using fewer ResNet layers to reduce overfitting on small data

---

## 📌 Summary

This project served as a foundational attempt to apply **PANet-style segmentation** to medical ultrasound images. While the current implementation is incomplete and underperforms, it provides the following:

- ✅ A working dataset pipeline
- ✅ Integration of pretrained ResNet features
- ⚠️ A placeholder model that mimics the general direction of PANet
- ❌ Lacks critical components of PANet that are essential for success

---

## 📎 Final Verdict

> The model does not currently perform well on BUS-UCLM and should **not** be considered production-ready. However, the work establishes a modular structure that can be iteratively improved into a full PANet implementation for medical segmentation tasks.

