# 📘 MultiResUNet: Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation

**Authors**: Nabil Ibtehaz, M. Sohel Rahman  
**Affiliations**: Samsung R&D Institute Bangladesh, BUET CSE Department  
**Published in**: Neural Networks, Volume 121, 2020, Pages 74–87  
**DOI**: [10.1016/j.neunet.2019.08.025](https://doi.org/10.1016/j.neunet.2019.08.025)

---

## 🧠 Overview

MultiResUNet is a powerful and improved version of the U-Net architecture built for **multimodal biomedical image segmentation**. It handles challenging cases such as:

- **Different object sizes** (multi-scale)
- **Faint/unclear boundaries**
- **Presence of noise, outliers, artifacts**
- **Large semantic gap between encoder and decoder features**

### 🔧 Core Improvements Over U-Net:
- ✅ **MultiRes Blocks** (multi-scale convolutions with residuals)
- 🔄 **ResPaths** (residual skip connections to reduce semantic gap)
- ⚡ Better accuracy with fewer parameters and faster convergence

---

## 📐 Architecture Details

### 🔹 MultiRes Block

- Combines features from different convolution kernel sizes (3×3, 5×5, 7×7) using **stacked 3×3 conv layers** to approximate larger kernels.
- Gradually increases filters (W/6, W/3, W/2) and includes residual connections for better learning.
- Enhances **multi-resolution feature extraction**.

### 🔹 ResPath

- Added on skip connections (instead of plain concat).
- Applies convolutional layers + residual connections.
- Helps align low-level encoder features with high-level decoder features.
- Number of conv blocks in ResPaths: 4 → 3 → 2 → 1 (outside → inside).

### 📊 Filter Configuration

| Block            | Filter Sizes           |
|------------------|------------------------|
| MultiRes Blocks  | 3×3 with W-based filter split |
| ResPaths         | 3×3 conv + 1×1 residual conv |
| Activations      | ReLU (all except last layer) |
| Output Layer     | 1×1 conv + Sigmoid      |
| Normalization    | Batch Normalization     |

---

## 🗂️ Datasets Used

| Modality               | Dataset          | # Images      | Resolution        | Input Size     |
|------------------------|------------------|---------------|--------------------|----------------|
| Fluorescence Microscopy| Murphy Lab       | 97            | 1344×1024          | 256×256        |
| Electron Microscopy    | ISBI-2012        | 30            | 512×512            | 256×256        |
| Dermoscopy             | ISIC-2018        | 2594          | variable           | 256×192        |
| Endoscopy              | CVC-ClinicDB     | 612           | 384×288            | 256×192        |
| MRI (3D)               | BraTS17          | 285 volumes   | 240×240×155        | 80×80×48       |

---

## 📈 Experimental Setup

- **Framework**: Python 3, Keras with TensorFlow backend
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam (150 epochs)
- **Evaluation Metric**: Jaccard Index (Intersection over Union)
- **Cross-Validation**: 5-Fold
- **Preprocessing**: Image resizing, normalization ([0, 1] range)
- **Postprocessing**: Threshold of 0.5 on output
- **Hardware**: NVIDIA TITAN XP GPU (12 GB), 16 GB RAM

---

## 🧪 Results Summary

| Modality     | MultiResUNet (%) | U-Net (%) | Relative Improvement (%) |
|--------------|------------------|-----------|---------------------------|
| Dermoscopy   | 80.30 ± 0.37     | 76.43     | **+5.07**                 |
| Endoscopy    | 82.06 ± 1.59     | 74.50     | **+10.15**                |
| Fluorescence | 91.65 ± 0.95     | 89.30     | **+2.63**                 |
| Electron Microscopy | 87.95 ± 0.77 | 87.41     | **+0.62**                 |
| MRI (3D)     | 78.19 ± 0.78     | 77.11     | **+1.41**                 |

- MultiResUNet consistently outperforms U-Net.
- Especially better on **difficult images**: vague boundaries, noise, artifacts.
- Faster convergence and lower standard deviation.

---

## 🔍 Ablation Study

| Model Variant        | Avg Jaccard (%) on CVC-ClinicDB |
|----------------------|----------------------------|
| U-Net                | 74.5                      |
| Only ResPath         | 75.85                     |
| Only MultiRes Block  | 81.72                     |
| Full MultiResUNet    | **82.06**                 |

### ✅ Observations:
- MultiRes Block improves **boundary detection**
- ResPath improves **homogeneity and region continuity**
- Together, they work best

---

## 🧪 Robustness Evaluation

- **Handles vague boundaries** better (e.g. endoscopy, dermoscopy)
- **More immune to perturbations**: noise, artifacts, irregular textures
- **Rejects outliers** more effectively (e.g., debris in microscopy)
- **Segments majority class** without over-segmentation (e.g., EM images)

---

## 🎨 Visual Insights

- U-Net performs well on perfect images.
- MultiResUNet shows **clearer, more continuous segmentations** in tough scenarios.
- Better **boundary sharpness**, **object continuity**, and **outlier rejection**.

---

## 📊 Data Augmentation Impact

| Model        | Without Aug (%) | With Aug (%) | Gain (%) |
|--------------|------------------|---------------|----------|
| U-Net        | 74.50            | 79.24         | +4.74    |
| MultiResUNet | 82.06            | 84.97         | +2.91    |

---

## 🧠 Conclusion

- **MultiResUNet = U-Net + MultiRes Blocks + ResPaths**
- Designed for **better scale awareness**, **semantic alignment**, and **robustness**.
- Outperforms U-Net across multiple challenging datasets and tasks.
- Performs better even in **3D MRI scans**.

### 💡 Future Work:
- Explore better loss functions (e.g., Dice loss)
- Extend experiments with more datasets
- Try domain-specific pre/post processing
- Fine-tune hyperparameters more aggressively

---

## 📎 Citation

```bibtex
@article{ibtehaz2020multiresunet,
  title={Multiresunet: Rethinking the U-net architecture for multimodal biomedical image segmentation},
  author={Ibtehaz, Nabil and Rahman, M Sohel},
  journal={Neural Networks},
  volume={121},
  pages={74--87},
  year={2020},
  publisher={Elsevier}
}
```
# 🧠 MultiResUNet: Simple Explanation for Medical Image Segmentation

This is a simplified summary of the paper:  
**“MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical image segmentation”**  
By Nabil Ibtehaz and M. Sohel Rahman  
Published in *Neural Networks, 2020*  
[Read the full paper →](https://doi.org/10.1016/j.neunet.2019.08.025)

---

## 📌 What Is This Paper About?

The authors noticed that **U-Net**, a very popular model for medical image segmentation, doesn't always work well when:

- The image contains **objects of very different sizes**
- **Boundaries are faint or hard to see**
- There is **noise or random clutter**
- The **encoder and decoder don’t “understand” each other well**

So, they created **MultiResUNet** — a better version of U-Net that can deal with these challenges.

---

## 🧩 What’s New in MultiResUNet?

### 1. 🔍 MultiRes Block
- Looks at the image using different zoom levels (like using different filter sizes).
- Helps detect small and large features better.
- Combines multiple small filters (3×3) to imitate large filters (5×5, 7×7) without using too much memory.

### 2. 🔄 ResPath
- Skip connections in U-Net are basic. They just copy-paste features from encoder to decoder.
- But this causes confusion. Why? Early layers see low-level details (edges), later layers see high-level meaning (objects).
- ResPath adds extra layers (with residual connections) to help the encoder and decoder communicate better.

---

## 🏥 What Data Did They Use?

They tested the model on **5 different types of medical images**:

| Type                  | Dataset         | Description                               |
|-----------------------|------------------|-------------------------------------------|
| Microscopy (fluorescence) | Murphy Lab       | Tiny cells with brightness variation      |
| Electron Microscopy   | ISBI-2012        | Very detailed nerve cord images           |
| Skin (dermoscopy)     | ISIC-2018        | Skin lesion images                        |
| Colonoscopy           | CVC-ClinicDB     | Polyps with blurry boundaries             |
| Brain MRI (3D)        | BraTS17          | Tumor detection in 3D MRI scans           |

---

## 🧪 Key Results

MultiResUNet beats U-Net in all datasets. It’s especially better when:

- Boundaries are blurry
- Background is noisy
- The object is tiny or oddly shaped

| Dataset      | U-Net (%) | MultiResUNet (%) | Improvement |
|--------------|-----------|------------------|-------------|
| Endoscopy    | 74.50     | **82.06**        | +10.15%     |
| Dermoscopy   | 76.43     | **80.30**        | +5.07%      |
| Microscopy   | 89.30     | **91.65**        | +2.63%      |
| MRI (3D)     | 77.11     | **78.19**        | +1.41%      |
| EM (majority class) | 87.41 | **87.95**    | +0.62%      |

---

## ⚙️ Training Details

- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam (150 epochs)
- **Evaluation Metric**: Jaccard Index (IoU)
- **Validation**: 5-Fold Cross-Validation
- **Preprocessing**: Resize + normalize to [0, 1]
- **Postprocessing**: Threshold at 0.5

---

## 💥 Why MultiResUNet Is Better

- **Faster training** and better results
- Detects **boundaries more accurately**
- Handles **irregular textures and background noise**
- More **reliable in tricky cases** with faint contrast or outliers
- **Works well even without data augmentation**, but improves with it

---

## 🔍 Ablation Study (What matters more?)

| Model Variant        | Performance (CVC) |
|----------------------|-------------------|
| U-Net                | 74.5%             |
| Only ResPath         | 75.9%             |
| Only MultiRes Block  | 81.7%             |
| Full MultiResUNet    | **82.1%**         |

Conclusion: **Using both components together gives the best result**.

---

## ✨ Takeaways

- MultiResUNet = U-Net + smarter blocks + better connections
- Works well even in hard cases with noise, weak boundaries, and complex textures
- Should be considered as a **better alternative to classical U-Net**
- Easy to adapt to 3D segmentation problems like brain tumor detection

---

## 📎 Citation

```bibtex
@article{ibtehaz2020multiresunet,
  title={Multiresunet: Rethinking the U-net architecture for multimodal biomedical image segmentation},
  author={Ibtehaz, Nabil and Rahman, M Sohel},
  journal={Neural Networks},
  volume={121},
  pages={74--87},
  year={2020},
  publisher={Elsevier}
}
```
