# DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models

**Authors:** Haoyang Li, Liang Wang, Chao Wang*, Jing Jiang, Yan Peng*, Guodong Long*  
(* denotes corresponding authors)  
**Institutions:** Shanghai University, University of Technology Sydney  
**Paper:** [CVPR 2025](https://github.com/JREion/DPC)  

---

## 📌 Overview

This repository contains the implementation of **DPC** — **Dual-Prompt Collaboration**, a plug-and-play framework designed to solve the **Base-New Trade-off (BNT)** problem in CLIP-based prompt tuning.

> **BNT Problem**: When fine-tuning on base (training) classes, the model often overfits and loses performance on new (unseen) classes.

### 🔍 What’s New in DPC?
- **Prompt-Level Decoupling**: Two separate prompts:
  - **Tuned Prompt (P)** → Maintains generalization for new classes.
  - **Parallel Prompt (P′)** → Optimized for base classes.
- **Weighting-Decoupling Module**: Dynamically controls each prompt’s influence.
- **Dynamic Hard Negative Optimizer (DHNO)**: Creates harder examples for stronger base class learning.
- **Feature Channel Invariance**: Keeps feature distribution stable during optimization.

---

## 📚 Abstract (Simplified)

In CLIP-based prompt tuning, focusing too much on base classes hurts generalization to new classes — the **BNT problem**.  
Existing methods try to balance this using the **same prompt** for both tasks, causing conflicting optimization directions.

**DPC** solves this by:
1. **Duplicating the prompt** into two: one frozen for generalization, one optimized for base tasks.
2. Using **Weighting-Decoupling** to adjust their influence independently.
3. Applying a **Dynamic Hard Negative Optimizer** to strengthen base class learning.

**Result:** Better base performance **without sacrificing** new class generalization — and no external data required.

---

## 🏗 Architecture

### Existing Prompt Tuning
One prompt → Optimized for both base and new → Conflicting gradients → BNT problem.

shell
Copy
Edit

### DPC Approach
Tuned Prompt (P) → frozen for new classes
Parallel Prompt (P′) → optimized for base classes
Weighting-Decoupling → balances them during inference
Dynamic Hard Negative Optimizer → pushes P′ harder

yaml
Copy
Edit

---

## ⚙️ Method

### 1️⃣ Dual Prompt Initialization
- Fine-tune the backbone prompt learner to get **Tuned Prompt P**.
- Clone it into **Parallel Prompt P′** for base-specific optimization.

```python
P_prime = P.clone()
2️⃣ Dynamic Hard Negative Optimizer (DHNO)
Goal: Make the base prompt’s job harder → better learning.

Steps:

Negative Sampler – Use P to get top-K most similar wrong predictions for each base image.

Feature Filtering – L2 normalize text features to keep the original distribution stable.

Hard Negative Optimizing – Train with symmetric InfoNCE contrastive loss.

3️⃣ Weighting-Decoupling Module (WDM)
Purpose: Control influence of each prompt in inference.

Base classes:

Copy
Edit
P̃_b = ω_b * P′ + (1 - ω_b) * P
New classes:

Copy
Edit
P̃_n = ω_n * P′ + (1 - ω_n) * P
Where:

ω_b = base prompt weight (best ≈ 0.2)

ω_n = new prompt weight (best ≈ 1e-6)

📊 Experiments
Datasets
Base-to-New: ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, UCF101.

Cross-Domain: ImageNet-V2, ImageNet-Sketch, ImageNet-A, ImageNet-R.

Results
On 11 datasets, DPC improved base accuracy in all backbones without harming new-class performance.

Achieved SOTA harmonic mean (H) scores.

Outperformed DePT, another plug-and-play method.

🔬 Ablation Studies
Component	Base ↑	New	H ↑
Baseline (CoOp)	81.98	68.84	74.84
+ Two-Step	82.69	68.39	74.86
+ DHNO	84.28	64.12	72.83
+ Weighting-Decoupling	85.15	68.84	76.13

Key Takeaways:

Both DHNO and WDM are necessary.

Prompt-level decoupling avoids BNT entirely.

Still works well with fewer fine-tuning epochs.

🧠 Why It Works
Feature Channel Invariance: Feature filtering ensures the generalization prompt’s distribution is unchanged.

Prompt-Level Decoupling: Eliminates gradient conflicts that hurt new-class performance.

🚀 Usage
bash
Copy
Edit
# Clone repo
git clone https://github.com/JREion/DPC.git
cd DPC

# Install dependencies
pip install -r requirements.txt

# Train with DPC on CoOp backbone
python train_dpc.py --backbone coop --dataset imagenet --omega_b 0.2 --omega_n 1e-6 --top_k 8
📝 Plain Language Summary
Think of prompts like instructions given to CLIP.
Old methods used one instruction for both the exam you practiced for (base classes) and surprise questions (new classes). Editing it for one hurt the other.

DPC’s idea:

Two separate instructions:

One for base → trained aggressively.

One for new → kept safe.

Weighting-Decoupling decides how much each counts.

Hard Negative Optimizer is like a coach giving trickier practice questions.
