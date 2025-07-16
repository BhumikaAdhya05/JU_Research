# 🧠 PANet for Few-Shot Breast Tumor Segmentation (BUS-UCLM Dataset)

This project implements the **Prototype Alignment Network (PANet)** for **few-shot segmentation** on breast ultrasound tumor images using the **BUS-UCLM dataset**. This work was carried out as part of a research internship under **Professor Ram Sarkar** at **Jadavpur University**.

---

## 📌 Objective

Perform breast tumor segmentation using **few-shot learning** — i.e., segmenting tumors using only **1 annotated support image (1-shot)** per episode.

---

## 📂 Dataset

- **Name**: [BUS-UCLM Breast Ultrasound Dataset](https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset)
- **Total Samples**: 683 images
- **Classes**: benign, malignant, and normal
- **Annotations**: Pixel-wise binary segmentation masks
- **Modality**: Grayscale ultrasound
- **Format**: PNG images and corresponding binary masks
- **License**: CC BY 4.0

---

## 🧠 Model Architecture: PANet

PANet is a **prototype-based few-shot segmentation model**. It operates in two steps:
1. Compute a **prototype vector** from support image features masked by tumor region.
2. Match this prototype with the query image using **cosine similarity** and generate a **similarity map**, which is then decoded into a segmentation mask.

### 🔧 Modules

#### ✅ Encoder
- Pre-trained **ResNet-50** backbone
- Converts input image into deep feature maps
- Grayscale input repeated to 3 channels
- Output: `(B, C, H, W)`

#### ✅ Prototype Computation
- Uses support features and masks to compute a class prototype vector
- Prototype = **mean feature vector over foreground pixels**
- Features and prototype are **L2-normalized**

#### ✅ Cosine Similarity Map
- Cosine similarity between prototype and query features yields similarity map
- Map resized to `(256, 256)` for segmentation

#### ✅ Decoder
A simple 3-layer CNN:
```python
self.decoder = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 1, 1)
)
```

---

## 🛠️ Implementation Details

| Component           | Value |
|---------------------|--------|
| Framework           | PyTorch (Google Colab) |
| Pre-trained model   | ResNet-50 (ImageNet) |
| Few-shot setup      | 1-shot (support image + mask) |
| Loss functions      | Dice Loss + Binary Cross Entropy (BCE) |
| Optimizer           | Adam |
| Input resolution    | 256 × 256 |
| Training episodes   | 200 |

---

## 🔍 Debugging Logs & Observations

| Metric              | Observation |
|---------------------|-------------|
| **Prototype Norm**  | ✅ Fixed at ~1.0 after normalization |
| **Similarity Map**  | ✅ Range ~0.4 to 0.8 — normal |
| **Prediction Mean** | Started at ~0.45 → plateaued ~0.38 |
| **Visual Output**   | ❌ Predicted masks were mostly black |
| **Loss Curve**      | Fluctuated around 1.5 → 1.3, no deep convergence |

---

## ❌ What Went Wrong

### 1. Prototype Feature Masking Bug
- Original implementation applied mask with mismatched spatial dimensions.
- **Fix**: Interpolated support masks to match feature shape and applied channel-wise masking.

### 2. Weak Decoder
- Initial decoder lacked capacity to translate similarity maps into masks.
- **Fix**: Upgraded to a deeper decoder with two ReLU layers.

### 3. Class Imbalance
- Tumors occupy tiny region in most images → model biased to background.
- **Impact**: BCE loss dominated by background pixels, suppressing learning signal for tumors.

### 4. Low Prediction Activations
- `sigmoid(pred_mask).mean()` remained low (~0.38) through training.
- **Impact**: Predicted masks were too close to 0 — mostly black.

---

## 🧪 What We Tried

✅ Fixed mask-feature alignment using interpolation  
✅ Normalized features and prototype vectors  
✅ Added deeper decoder  
✅ Used BCE + Dice Loss  
✅ Visualized predictions every 20 episodes  
✅ Verified support/query masks had foreground pixels  
✅ Monitored similarity map and prototype norms  

---

## 🧩 Limitations

- PANet assumes strong support-query similarity — less likely in variable ultrasound tumor data.
- ResNet-50 (RGB) on grayscale images required channel repetition — not ideal for medical imaging.
- Decoder is shallow compared to UNet / DeepLabV3+.
- Lacks attention and multi-scale contextual understanding.
- Full-image training leads to class imbalance due to small tumor size.

---

## 🚀 Future Improvements

✅ **Switch PANet to Stronger Few-Shot Architectures**
- RePRI (Refined Prototypes)
- CyCTR (Transformer-based context refinement)
- FSS-1000 fine-tuning + meta-learning
- CLIP + Adapter for VLM-based segmentation

✅ **Enhance Decoder**
- Add skip connections or multi-scale fusion
- Integrate attention or transformers in decoding

✅ **Sampling Strategy**
- Crop tumor regions → improve foreground-background balance

✅ **Meta-Learning**
- Try MAML, Meta-SGD for task-adaptive learning

✅ **Data Augmentation**
- Use flips, rotations, ultrasound-specific noise injection

---

## ✅ Final Takeaway

We **successfully implemented PANet** for few-shot tumor segmentation on the BUS-UCLM dataset. Despite fixing major bugs in feature masking and normalizing prototypes, the model failed to produce meaningful segmentations. Prediction activations remained low, and the decoder struggled with learning discriminative masks.

However, this forms a **solid foundation** for exploring stronger few-shot and meta-learning-based segmentation approaches. The pipeline and debugging insights here will directly support follow-up work using more robust few-shot methods.

---


