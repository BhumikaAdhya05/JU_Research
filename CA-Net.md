# Multimodal Medical Image Segmentation using Multi-Scale Context-Aware Network (CA-Net)

## 📌 Overview

This repository implements **CA-Net**, a **Context-Aware U-Net-based architecture** for **multimodal medical image segmentation**. The network is designed to address challenges in segmenting objects of varying scales and irregular boundaries across **dermoscopy**, **CT**, and **retina** modalities.

---

## 📚 Highlights

- **Multi-Scale Context Fusion (MCF) Module**: Integrates spatial context and channel-wise attention.
- **Dense Skip Connections (DSC)**: Preserve detailed spatial information from multiple encoder levels.
- **Lightweight and Efficient**: Achieves better results than U-Net with fewer parameters and FLOPs.
- **Tested on Three Modalities**:
  - Skin lesion segmentation (ISIC 2018)
  - Lung segmentation (CT images)
  - Blood vessel segmentation (DRIVE dataset)

---

## 🔧 Network Architecture

CA-Net consists of:

1. **Encoder**: Standard U-Net style convolution + pooling layers.
2. **Multi-Scale Context Fusion (MCF)**:
   - **SCF Block** (Spatial Context Fusion):
     - Four parallel branches using atrous convolutions with dilation rates: `1`, `3`, `5`.
     - Receptive fields: `3x3`, `7x7`, `11x11`, `19x19`.
   - **SE Block** (Squeeze-and-Excitation):
     - Channel-wise attention with global average pooling and 1x1 convolutions.
   - **Residual Connection**: Aids in gradient flow and preserves context.
3. **Decoder**: Upsampling with dense skip connections.
4. **Dense Skip Connections**:
   - Fuse features from current and **all previous encoder levels**.
   - Downsample with MaxPool and match channels with 1x1 Conv before addition.

---

## 🔬 Datasets

| Dataset       | Modality     | Images | Original Res | Input Res   |
|---------------|--------------|--------|--------------|-------------|
| ISIC 2018     | Dermoscopy   | 2594   | variable     | 448 × 448   |
| Lung Dataset  | CT           | 267    | 512 × 512    | 512 × 512   |
| DRIVE         | Retina       | 40     | 565 × 584    | 256 × 256   |

---

## 📊 Evaluation Metrics

- **Accuracy (Acc)**  
- **Sensitivity (Sen)**  
- **Specificity (Spec)**  
- **Precision (Prec)**  
- **F1 Score**

---

## 🧪 Results

### ✅ ISIC 2018 (Skin Lesion Segmentation)

| Model         | F1 Score | Acc   | Sen   | Spec  |
|---------------|----------|-------|-------|-------|
| U-Net         | 0.647    | 0.890 | 0.708 | 0.964 |
| Atten U-Net   | 0.665    | 0.897 | 0.717 | 0.967 |
| R2U-Net       | 0.679    | 0.880 | 0.792 | 0.928 |
| BCDU (d=3)    | 0.851    | 0.937 | 0.785 | 0.982 |
| **CA-Net**    | **0.868**|**0.957**|**0.855**|**0.985**|

### ✅ Lung CT Segmentation

| Model         | F1 Score | Acc   | Sen   | Overlap Error (E) |
|---------------|----------|-------|-------|--------------------|
| U-Net         | 0.875    | 0.939 | 0.974 | 0.139              |
| Atten U-Net   | 0.980    | 0.991 | 0.982 | 0.024              |
| CE-Net        | -        | 0.990 | 0.980 | 0.038              |
| **CA-Net**    | **0.981**|**0.992**|**0.983**|**0.023**         |

### ✅ DRIVE (Retina Vessel Segmentation)

| Model         | F1 Score | Acc   | Sen   | Spec  |
|---------------|----------|-------|-------|-------|
| U-Net         | 0.8142   | 0.9531| 0.7537| 0.9820|
| BCDU (d=3)    | 0.8224   | 0.9560| 0.8007| 0.9786|
| **CA-Net**    | **0.8254**|**0.9561**|0.7934|**0.9812**|

---

## 🔍 Ablation Study

| Configuration                         | ISIC F1 | Lung F1 | DRIVE F1 |
|--------------------------------------|---------|---------|----------|
| U-Net                                | 0.647   | 0.875   | 0.8142   |
| U-Net + MCF                          | 0.853   | 0.981   | 0.8168   |
| U-Net + MCF + Atrous                 | 0.865   | 0.967   | 0.8230   |
| U-Net + Dense Skip Connections (DSC) | 0.830   | 0.979   | 0.8165   |
| **CA-Net (MCF + Atrous + DSC)**      | **0.868**|**0.981**|**0.8254**|

---

## 🧠 Model Complexity

| Model                        | Params (M) | FLOPs (G) |
|-----------------------------|------------|-----------|
| U-Net                       | 31.04      | 167.56    |
| CA-Net                      | **23.50**  | **166.28**|

---

## 🛠 Training Details

- **Framework**: PyTorch  
- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Adam  
- **Learning Rate**: 0.0001, decayed by 0.95 if plateaued for 5 epochs  
- **Early Stopping**: LR < 1e-6  
- **Epochs**: 200  
- **Batch Size**: ISIC: 6 | Lung: 4 | DRIVE: 6  
- **Data Augmentation**: (DRIVE) Rotation, flip, zoom, crop, brightness adjustment

---

## 🧑‍💻 Authors

- **Xue Wang** – Methodology, Software, Writing  
- **Zhanshan Li** – Conceptualization, Writing  
- **Yongping Huang** – Resources, Supervision  
- **Yingying Jiao** – Methodology, Writing

---

## 💡 Future Work

- Introduce **weighted attention in skip connections**.
- Explore **channel- and scale-wise fusion** more deeply.

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@article{wang2022canet,
  title={Multimodal medical image segmentation using multi-scale context-aware network},
  author={Wang, Xue and Li, Zhanshan and Huang, Yongping and Jiao, Yingying},
  journal={Neurocomputing},
  volume={486},
  pages={135--146},
  year={2022},
  publisher={Elsevier}
}
```

# 📘 Simple Explanation: CA-Net for Medical Image Segmentation

This is a simplified summary of the research paper:  
**“Multimodal medical image segmentation using multi-scale context-aware network (CA-Net)”**  
Published in *Neurocomputing 2022 (Vol. 486)*

---

## 🧠 What Is This Paper About?

This paper introduces **CA-Net**, a smart deep learning model used to automatically identify and segment medical images. It works well across different types of scans like:

- Skin lesion images (e.g., for cancer detection)
- CT scans of lungs
- Retina (eye) images

---

## 🚀 Why Is It Important?

Segmenting medical images is **hard** because:

- Lesions or organs come in **different shapes and sizes**
- They may only occupy a **tiny part** of the image
- The boundaries are often **blurry or unclear**

Traditional methods or even the standard **U-Net** model miss some of this important context. That’s where CA-Net helps.

---

## 🧩 What Is CA-Net Made Of?

CA-Net is based on U-Net but improves it using two smart ideas:

### 1. 🧭 Multi-Scale Context Fusion (MCF) Module
- **SCF block**: Looks at the image using different “zoom levels” (3x3, 7x7, etc.) to get more context.
- **SE block**: Decides which channels (features) are important and boosts them.

### 2. 🔗 Dense Skip Connections (DSC)
- Instead of connecting only matching layers in encoder-decoder like U-Net, CA-Net connects **ALL earlier layers**, so the decoder gets both low-level details and high-level meanings.

---

## 📊 How Well Does It Work?

CA-Net was tested on 3 medical datasets and outperformed several well-known models:

| Dataset      | Task               | Best F1 Score |
|--------------|--------------------|---------------|
| ISIC 2018    | Skin lesion        | 0.868         |
| Lung CT      | Lung segmentation  | 0.981         |
| DRIVE        | Retinal vessels    | 0.8254        |

CA-Net also had **fewer parameters and used less computing power** than some other models!

---

## ⚙️ Training Setup

- **Framework**: PyTorch  
- **Loss**: Binary Cross Entropy  
- **Optimizer**: Adam  
- **Epochs**: 200  
- **Batch Sizes**: 4–6  
- **Image Augmentations**: Rotate, flip, zoom, brightness

---

## 🧪 What Did the Experiments Show?

- Adding the **MCF module** helped CA-Net understand more about shape and structure.
- Adding **dense skip connections** helped it recover better image details.
- Combining both gave the best results with **less computation**.

---

## 🧠 Future Ideas

The authors want to:
- Add **weights** to skip connections (so the model learns which past info is more useful).
- Use CA-Net on **more types** of medical images.

---

## 📚 Citation (BibTeX)

```bibtex
@article{wang2022canet,
  title={Multimodal medical image segmentation using multi-scale context-aware network},
  author={Wang, Xue and Li, Zhanshan and Huang, Yongping and Jiao, Yingying},
  journal={Neurocomputing},
  volume={486},
  pages={135--146},
  year={2022},
  publisher={Elsevier}
}
```
