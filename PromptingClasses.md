# Prompting Classes: Exploring the Power of Prompt Class Learning in Weakly Supervised Semantic Segmentation

**Authors:** Balamurali Murugesan\*, Rukhshanda Hussain\*, Rajarshi Bhattacharya\*, Ismail Ben Ayed, Jose Dolz
**Affiliations:** ETS Montreal, Jadavpur University
**Paper:** WACV 2024
**Code:** [https://github.com/Ruxie189/WSS\_POLE](https://github.com/Ruxie189/WSS_POLE)

> *Rukhshanda Hussain and Rajarshi Bhattacharya did this work as part of a research internship at ETS Montreal.*

---

## 🔍 Abstract

* Vision-language pretraining models like CLIP show strong generalization.
* Prompt tuning is effective for downstream adaptation.
* This work studies **prompt tuning for Weakly Supervised Semantic Segmentation (WSSS)**.
* Key observations:

  * Modifying only the `[CLS]` token (category name) greatly impacts CAM quality.
  * Optimizing `[CTX]` (context) tokens has lesser effect than tweaking `[CLS]`.
  * Best CAMs often come from **semantically similar synonyms**, not the ground truth class label.
* Propose **POLE** (PrOmpt cLass lEarning):

  * Learns best-correlated `[CLS]` token from candidate set.
  * Significantly improves CAM quality and segmentation.

---

## ✍️ Introduction

* **Semantic Segmentation**: Fundamental CV task, needs pixel-level labels.
* **WSSS**: Uses cheaper labels like image tags, bounding boxes, scribbles.
* WSSS relies on **CAMs from classifiers**, refined into pseudo-labels.
* Issue: CAMs focus on discriminative parts, miss full object extent.
* Prior fixes: Erasing/mining regions, attention, equivariant constraints, iterative methods.
* But: Still suboptimal, complex pipelines.

### Vision-Language Models to the Rescue

* **CLIP** links vision and language in a shared space.
* **Prompt Design** (text input to CLIP) plays a huge role in downstream performance.
* Prior work focused on context tokens `[CTX]`, ignored category token `[CLS]`.
* This work investigates: How much does changing `[CLS]` help?

### Key Contributions

1. Show prompt design (esp. `[CLS]`) impacts CAMs and WSSS performance.
2. Ground truth class often **not** the best prompt.
3. Propose **POLE**: learns the `[CLS]` token that maximizes text-image correlation.
4. Achieves SOTA results on PASCAL VOC 2012 with fewer complexities.

---

## 📅 Related Work

### Weakly Supervised Segmentation (WSSS)

* CAMs localize object regions from classifiers.
* CAMs are refined using region mining, attention, saliency, equivariance, etc.

### CLIP-based Segmentation

* CLIMS \[CVPR 2022] used CLIP for WSSS, showed improved CAMs.
* Authors here improve further by analyzing prompt design.

### Prompt Learning

* Hot topic in NLP and VL tasks.
* Prior work mostly optimized `[CTX]` (continuous vectors), fixed `[CLS]`.
* This work focuses on choosing best `[CLS]` synonym instead of learning continuous embeddings.

---

## ⚖️ Methodology

### Problem Setup

* Given dataset $D = \{(X_i, y_i)\}_{i=1}^N$ where $X_i$: image, $y_i$: image-level multi-labels.
* Goal: Learn segmentation model from image-level labels only.

### CAM Generation

* Use ResNet-50 to extract features $Z \in \mathbb{R}^{C \times H \times W}$.
* Classifier: 1x1 conv + sigmoid.
* CAM for class $k$: $P_k(h, w) = \sigma(W_k^T Z(h, w))$

### Prompt Class Learning (POLE)

**Step 1: Generate Similar Class Names**

* Use ChatGPT or corpus (Wikipedia, BNC, etc.) to get synonyms for each class.

**Step 2: Compute Text Embeddings**

* Use CLIP text encoder to get $v^t_{k}$ for each synonym prompt.

**Step 3: Compute Image Embedding**

* Multiply input image with CAM $X \cdot P_k$ to focus on target regions.
* Compute visual embedding $v^i_k$ using CLIP image encoder.

**Step 4: Match & Select Best `[CLS]`**

* Use cosine similarity: $sim(v^i_k, v^t_{kj})$
* Choose class $CLS^*$ with highest similarity.

### Contrastive Loss

```math
\mathcal{L}_{Cont} = -\alpha \sum y_k \log(s^o_k) - \beta \sum y_k \log(1 - s^b_k)
```

* $s^o_k$: similarity between target region and its text embedding.
* $s^b_k$: similarity between background and text embedding.

### Adapters

* Introduce lightweight adapters $A_v(\cdot), A_t(\cdot)$ (MLPs) to refine embeddings.
* Learnable mixing parameters $r_v, r_t$ for image and text.

---

## 📊 Experiments

### Dataset

* **PASCAL VOC 2012** + SBD (augmented): 1,464 train, 1,449 val, 1,456 test.
* Evaluated using **mean IoU (mIoU)**.

### Implementation

* Backbone: ResNet-50 (for CAMs).
* Optimizer: SGD + cosine decay.
* Training: 10 epochs, batch size 16.
* Initial CAMs refined using IRNet \[Ahn et al.].

### Results

#### Does prompt matter?

* Yes. Manual prompt: "A photo of \[CLS]" vs "An image of \[CLS]" changes mIoU.
* Optimizing `[CLS]` via synonyms beats all other prompt methods.

| Method      | CAMs     | +RW      |
| ----------- | -------- | -------- |
| CLIMS \[66] | 56.6     | 70.5     |
| CoOp \[76]  | 57.6     | 73.1     |
| DeFo \[58]  | 56.6     | 73.2     |
| POLE (Ours) | **59.0** | **74.2** |

#### Comparison with SOTA

* POLE outperforms or matches all recent WSSS models on PASCAL VOC 2012.

#### Synonym Source Matters

* Best performance comes from ChatGPT synonyms.
* More synonyms = better selection = better CAMs.

#### Adapters Help

| Setting              | mIoU (%) |
| -------------------- | -------- |
| Ground truth `[CLS]` | 70.5     |
| Optimized `[CLS]*`   | 73.6     |
| + Fixed adapters     | 73.8     |
| + Learnable adapters | **74.2** |

---

## 🔝 Conclusion

* Prompt tuning, especially of the `[CLS]` token, is powerful for WSSS.
* Using a synonym (not the ground truth label) can yield **better CAMs**.
* **POLE**: simple but effective framework that selects the best prompt class.
* Achieves **SOTA** on a popular benchmark with minimal architectural changes.

---

## 🔍 Citation

```bibtex
@inproceedings{murugesan2024prompting,
  title={Prompting Classes: Exploring the Power of Prompt Class Learning in Weakly Supervised Semantic Segmentation},
  author={Murugesan, Balamurali and Hussain, Rukhshanda and Bhattacharya, Rajarshi and Ben Ayed, Ismail and Dolz, Jose},
  booktitle={WACV},
  year={2024}
}
```

---

## 💡 Keywords

`Prompt Learning`, `Weakly Supervised Semantic Segmentation`, `CLIP`, `Vision-Language Models`, `Few-shot`, `Semantic Segmentation`, `POLE`, `Prompt Optimization`

---

## 📃 References

Key prior works cited:

* CLIMS \[CVPR 2022]
* CoOp \[CVPR 2022]
* DeFo \[2022]
* IRNet \[CVPR 2019]
* Zhou et al. \[CLIP Prompting]

(See full reference list in paper)

---

*This markdown is structured for GitHub rendering. Paste into your README or documentation directly.*
