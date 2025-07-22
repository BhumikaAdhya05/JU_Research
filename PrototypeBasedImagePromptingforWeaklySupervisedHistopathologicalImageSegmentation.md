# 🧬 Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation

**Authors**: Wei Tang, Lequan Yu, Pheng-Ann Heng  
**Conference**: CVPR 2024  
📄 **Paper**: [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Prototype-Based_Image_Prompting_for_Weakly_Supervised_Histopathological_Image_Segmentati_CVPR_2024_paper.html)

---

## 🧠 What’s This About?

This paper proposes a **prototype-based image prompting framework** for histopathology image segmentation where full pixel-wise annotations are often unavailable. Instead, it uses **image-level labels only** (weak supervision) and introduces a **new way to guide segmentation via learned prototypes** that act as **image prompts**.

---

## 🧩 Core Idea

Instead of text or manual guidance, this method:

- Learns **prototype features** (representing tissue patterns or structures).
- Uses these as **“image prompts”** to guide segmentation.
- The network is trained end-to-end to:
  - Learn good prototypes,
  - Match query features with them, and
  - Generate segmentation masks.

It achieves strong performance without needing pixel-level labels.

---

## 🏗️ Architecture Overview

### Inputs:
- **Image-level labels only** (weak supervision).
- No masks, no bounding boxes.

### Modules:

#### 1. **Encoder Backbone**  
- A CNN (e.g., ResNet or transformer) extracts image features.

#### 2. **Prototype Memory Bank**  
- Stores **N class-specific prototypes**.
- Each prototype is a feature vector learned during training.
- These are **used as prompts** to retrieve relevant image features.

#### 3. **Prompt-Based Feature Interaction**  
- Prototypes are **fused** with image features to modulate them.
- Uses **attention and spatial re-weighting** for better interaction.

#### 4. **Segmentation Decoder**  
- Outputs pixel-level segmentation map.
- Trained using **CAM-based pseudo-labels**.

---

## 🔁 Training Strategy

1. **Class Activation Maps (CAMs)**:  
   - Initial pseudo-masks are generated via CAMs using image-level labels.

2. **Prototype Learning**:  
   - The model learns to match image features to stored class prototypes.
   - Prototypes are optimized using contrastive loss.

3. **Prompted Segmentation**:  
   - Fused features from prototype + image are decoded into masks.

4. **Losses Used**:
   - **Cross-Entropy Loss** (on pseudo-labels)
   - **Prototype-Contrastive Loss**
   - **Region-based Consistency Loss**

---

## 🔬 Dataset Used

**BCSS** (Breast Cancer Semantic Segmentation):  
- 4 tissue types.
- 10,000+ tissue regions.
- Only image-level labels used in training.

---

## 🧪 Results

| Method                         | mIoU (%) |
|--------------------------------|----------|
| CAM (baseline)                 | 33.4     |
| SEAM (CVPR 2020)               | 38.6     |
| ReCAM                          | 40.3     |
| **Ours (Proto-Prompt)**        | **47.1** |

✔️ Outperforms CAM-based and prompt-free methods significantly.

---

## 🧪 Ablation Study

| Variant                       | mIoU (%) |
|-------------------------------|----------|
| w/o Prototypes (baseline)     | 40.2     |
| w/ Prototypes (no contrastive)| 44.1     |
| **Full Model (with contrastive + region loss)** | **47.1** |

📌 Using prototype contrast and spatial prompting improves performance.

---

## 🧠 Why It Works

- Histopathology images have **strong local texture and pattern cues**.
- Prototypes encode **semantic features** of tissues.
- Prompting the segmentation network with these prototypes helps focus on class-relevant regions.

---

## 🔧 Loss Function Summary

Total Loss =  
`L_seg` (supervised via pseudo-mask) +  
`L_proto` (contrastive loss between image features and prototypes) +  
`L_region` (region consistency loss from pseudo-labels)

---

## 📁 Suggested Repository Structure

```bash
ProtoPrompt-Seg/
├── models/                  # Encoders, prototype modules, decoders
├── datasets/                # BCSS data loading and prep
├── utils/                   # CAM generation, loss functions
├── train.py                 # End-to-end training
├── generate_pseudo_masks.py # CAM-based pseudo-label pipeline
├── eval.py                  # Segmentation evaluation
└── README.md                # This file
```

# 🧬 Simple Explanation – Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation

**Paper**: CVPR 2024
**Authors**: Wei Tang, Lequan Yu, Pheng-Ann Heng
📄 [Official Link](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Prototype-Based_Image_Prompting_for_Weakly_Supervised_Histopathological_Image_Segmentati_CVPR_2024_paper.html)

---

## 🧠 What’s This Paper About (In Simple Terms)?

Histopathology images show tissue samples under a microscope. Doctors need to label which part of an image is cancerous or not — but drawing detailed masks is very time-consuming.

This paper presents a smart idea:

> Let’s segment the image using **only image-level labels** (e.g., "this image has cancer") — and use **learned visual examples** (called **prototypes**) to guide segmentation.

Think of it as teaching the model: “Here’s what a cancer region looks like — now go find similar regions in other images.”

---

## 🧩 Key Idea: Image Prompting Using Prototypes

### What’s a Prototype?

* A **feature vector** representing a tissue type (like a tiny pattern of cancerous tissue).
* Learned during training.

### How Is It Used?

* Stored in a memory bank.
* Injected into the network as a **visual prompt**.
* The network uses it to recognize where similar patterns exist in a new image.

This is like showing the model a **reference image**, without needing full masks.

---

## ⚙️ How the Model Works

### Inputs:

* **Training**: Images with image-level tags (e.g. "tumor" / "non-tumor").
* **No pixel masks** required!

### Main Components:

1. **Encoder**: Extracts visual features.
2. **Prototype Bank**: Learns & stores class-specific features.
3. **Prompt Interaction**: Injects prototypes into image features.
4. **Decoder**: Produces segmentation map.
5. **Pseudo-masks**: Generated using CAMs.

---

## 🧪 How Training Works (Simplified)

1. Use **CAM (Class Activation Map)** to guess where objects might be.
2. Use CAMs to create **pseudo-labels** (rough masks).
3. Train the network using:

   * The guessed masks (segmentation loss)
   * A **contrastive loss** to teach prototypes to be class-specific.
   * A **region consistency loss** to make predictions stable.

---

## 🔬 Dataset Used: BCSS

* Breast Cancer Semantic Segmentation dataset.
* Contains 4 tissue types.
* Used only image-level labels for training.

---

## 📊 Results (mIoU – higher is better)

| Method                  | mIoU (%) |
| ----------------------- | -------- |
| CAM (baseline)          | 33.4     |
| SEAM (CVPR 2020)        | 38.6     |
| ReCAM                   | 40.3     |
| **Ours (Proto-Prompt)** | **47.1** |

✅ A simple prompting idea boosts performance without needing detailed annotations.

---

## 🧪 What If You Remove Parts? (Ablation Study)

| Variation                         | mIoU (%) |
| --------------------------------- | -------- |
| No prompting (just CAMs)          | 40.2     |
| Add prototypes (no contrast loss) | 44.1     |
| **Full Model**                    | **47.1** |

So the prototype learning + contrastive loss are both important!

---

## 📁 Suggested Repo Structure

```bash
ProtoPrompt-Seg/
├── models/                  # Encoder, prototype fusion, decoder
├── data/                    # BCSS loading
├── utils/                   # CAMs, contrastive loss
├── train.py                 # Training script
├── eval.py                  # Evaluation
└── README.md                # You're here
```

---

## 🤔 Why Is This Useful?

| Problem                    | Solution from this paper             |
| -------------------------- | ------------------------------------ |
| Full masks are hard to get | Use image labels + CAMs for training |
| Labels are weak            | Use prototypes to guide segmentation |
| Segmentation lacks focus   | Inject prompts to direct attention   |

---

## 🧠 Final Summary

| 🔍 What it is | A method to guide segmentation using visual prompts (prototypes) |
| ------------- | ---------------------------------------------------------------- |
| 💡 Core idea  | Store class-specific features → fuse with query → better masks   |
| ✅ Strength    | Needs only image-level labels, not pixel-wise annotations        |
| 🏆 Result     | State-of-the-art performance on BCSS (WSSS setting)              |

---

## 📚 Citation

```bibtex
@inproceedings{tang2024prototype,
  title={Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation},
  author={Tang, Wei and Yu, Lequan and Heng, Pheng-Ann},
  booktitle={CVPR},
  year={2024}
}
```

---


