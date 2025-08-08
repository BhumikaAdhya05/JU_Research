DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models
Authors: Haoyang Li, Liang Wang, Chao Wang*, Jing Jiang, Yan Peng*, Guodong Long*
(* denotes corresponding authors)
Institutions: Shanghai University, University of Technology Sydney
Paper: CVPR 2025

📌 Overview
This repository contains the implementation of DPC — Dual-Prompt Collaboration, a plug-and-play framework designed to solve the Base-New Trade-off (BNT) problem in CLIP-based prompt tuning.

BNT Problem: When fine-tuning on base (training) classes, the model often overfits and loses performance on new (unseen) classes.

🔍 What’s New in DPC?
Prompt-Level Decoupling: Instead of optimizing one prompt for both base and new tasks, DPC creates two separate prompts:

Tuned Prompt → Keeps generalization for new classes.

Parallel Prompt → Specially optimized for base classes.

Weighting-Decoupling: Dynamically control how much each prompt influences predictions.

Dynamic Hard Negative Optimizer (DHNO): Generates hard examples for stronger base class learning.

Feature Channel Invariance: The model keeps the feature space stable during optimization.

📚 Abstract (Simplified)
In CLIP-based prompt tuning, focusing on base classes improves performance there but hurts new class generalization (BNT problem).
Existing methods try to balance this with constraints but still optimize the same prompt for both tasks — leading to conflicts.

DPC solves this by:

Duplicating the prompt into two: one for base classes, one for new classes.

Using Weighting-Decoupling to control their influence independently.

Applying a Dynamic Hard Negative Optimizer to push the base-class prompt harder.

The result?
Better base performance without sacrificing new class generalization — and no external data needed.

🏗 Architecture
Existing Prompt Tuning
sql
Copy
Edit
One prompt → Optimized for both base and new tasks → Conflicting gradients → BNT problem.
DPC Approach
csharp
Copy
Edit
Tuned Prompt (P) → frozen for new class generalization
Parallel Prompt (P′) → trained for base class performance
Weighting-Decoupling → balances them
Dynamic Hard Negative Optimizer → pushes P′ harder
Figure 1 – Side-by-side comparison:
(a) Single prompt optimization (existing methods) vs. (b) DPC's dual prompt decoupling.

⚙️ Method
1️⃣ Dual Prompt Initialization
Start with a pretrained backbone (e.g., CoOp, MaPLe, PromptSRC, PromptKD).

Fine-tune to get Tuned Prompt P.

Clone it to create Parallel Prompt P′ for base class optimization.

python
Copy
Edit
P_prime = P.clone()
2️⃣ Dynamic Hard Negative Optimizer (DHNO)
Goal: Make base class optimization harder → stronger performance.

Steps:

Negative Sampler – Use tuned prompt P to find top-K most similar (but wrong) predictions for each image in base classes.

Feature Filtering – Keep feature distribution stable via L2 normalization before optimization.

Hard Negative Optimizing – Apply symmetric InfoNCE loss for contrastive learning.

3️⃣ Weighting-Decoupling Module (WDM)
Purpose: Control prompt influence during inference.

Base classes:

Copy
Edit
P̃_b = ω_b * P′ + (1 - ω_b) * P
New classes:

Copy
Edit
P̃_n = ω_n * P′ + (1 - ω_n) * P
Where:

ω_b = weight for base prompt (typically 0.2)

ω_n = weight for new prompt (close to 0)

📊 Experiments
Datasets
Base-to-New Generalization: ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, UCF101.

Cross-Dataset / Cross-Domain: ImageNet-V2, ImageNet-Sketch, ImageNet-A, ImageNet-R.

Results (Highlights)
On 11 datasets, DPC improved base class accuracy in all backbones without harming new class performance.

Achieved state-of-the-art Harmonic Mean (H) scores.

Outperformed another plug-and-play method DePT.

🔬 Ablation Studies
Component	Base ↑	New	H ↑
Baseline (CoOp)	81.98	68.84	74.84
+ Two-Step	82.69	68.39	74.86
+ DHNO	84.28	64.12	72.83
+ Weighting-Decoupling	85.15	68.84	76.13

Key findings:

Both DHNO and WDM are crucial.

ω_b = 0.2, ω_n ≈ 0 give best balance.

Even with fewer epochs, DPC outperforms baselines.

🧠 Why It Works
Feature Channel Invariance: DPC’s feature filtering ensures the new prompt’s feature distribution is preserved, so new-class performance doesn’t degrade.

Prompt-Level Decoupling: Avoids conflicting gradient updates that cause BNT.

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
📌 Citation
bibtex
Copy
Edit
@inproceedings{li2025dpc,
  title={DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models},
  author={Li, Haoyang and Wang, Liang and Wang, Chao and Jiang, Jing and Peng, Yan and Long, Guodong},
  booktitle={CVPR},
  year={2025}
}
📝 Explanation in Plain Language
Think of prompts like “instructions” given to CLIP.
Old methods wrote one instruction and kept editing it to be good at both the exam you practiced for (base classes) and surprise questions (new classes). This caused problems: improving one part made the other worse.

DPC says: “Why not have two separate instructions?”

One focuses on the practiced questions (base) → trained aggressively.

One stays safe for surprise questions (new) → frozen.

The Weighting-Decoupling module decides how much each “instruction” should count when answering.

And the Hard Negative Optimizer is like a coach that gives you trickier practice questions so you get even better on the base exam.

