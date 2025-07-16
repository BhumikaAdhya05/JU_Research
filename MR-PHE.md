# Leveraging Vision-Language Embeddings for Zero-Shot Learning in Histopathology Images (MR-PHE)

## Overview
This paper introduces **MR-PHE (Multi-Resolution Prompt-guided Hybrid Embedding)**, a novel zero-shot learning (ZSL) framework for histopathology image classification. It combines multiresolution visual cues with domain-specific prompt engineering to align visual and textual representations effectively without task-specific training.

## Motivation
Histopathology datasets are often unlabeled or under-annotated. ZSL aims to overcome this by leveraging pretrained vision-language models (VLMs) that can generalize to unseen classes using textual class descriptions. MR-PHE is designed to better capture the intricate visual-semantic relationships in histopathology by:
- Extracting multiscale image patches.
- Designing clinically relevant textual prompts.
- Fusing patch and global embeddings with hybrid weighting.

## Key Components

### 1. **Multiresolution Patch Extraction**
- Resize input image at multiple scales: `S = {0.25, 0.5, 0.75}`
- Extract `n = 5` random crops per scale.
- Include original image for global context.

### 2. **Visual Embedding**
- Use frozen image encoder `f` from the CONCH model.
- Compute normalized embeddings for all patches and the full image.

### 3. **Prompt Engineering**
- Create diverse textual prompts per class using:
  - Synonyms (e.g., “malignant”, “invasive”)
  - Templates (e.g., “H&E stained image of {class}”)
  - Clinical statements (e.g., “Tissue spreads to other parts…”)
- GPT-4 generates candidates; curated for domain relevance.

### 4. **Prompt Selection**
- Evaluate each prompt’s classification performance.
- Select top-K prompts (K=30) per class based on validation accuracy.

### 5. **Class Embedding Construction**
- Encode top prompts using text encoder `g` (also from CONCH).
- Weight prompt embeddings based on similarity to reference label.
- Construct final class embedding as a weighted average.

### 6. **Hybrid Embedding Fusion**
- Compute attention-weighted average of patch embeddings.
- Fuse with global embedding: `h = α * e_global + (1 - α) * e_patch`
- α=0.5 performs best.

### 7. **Classification**
- Compute cosine similarity between hybrid embedding `h` and class embeddings.
- Apply temperature-scaled softmax for final prediction.

## Results

### Zero-Shot Performance
MR-PHE outperforms prior ZSL baselines (CLIP, CuPL, WCA, etc.) on 6 public and private datasets:
- Up to **17.65% higher F1-score** than prior methods.
- Outperforms domain-specific foundation models (e.g., PLIP, BiomedCLIP).
- Competitive with state-of-the-art supervised methods.

### Ablation Study
Each module contributes significantly:
- Full MR-PHE: 96.71% accuracy (HE-GHI-DS dataset)
- Removing prompt selection or hybrid embedding reduces accuracy up to ~9%.

## Strengths
- **No need for retraining**—ZSL approach using frozen encoders.
- **Domain-aware** prompt design leads to better alignment.
- **Efficient**—real-time inference at >350 FPS.
- **Scalable** to multiple clinical settings with minimal annotation.

## Limitations
- Depends on quality of prompt curation.
- Doesn’t handle domain shifts like staining variations directly.
- Requires clinical validation for deployment.

## Conclusion
MR-PHE is a robust, modular, and clinically relevant ZSL framework tailored for histopathology. It blends vision-language alignment with pathology-specific domain insights to deliver strong classification performance without labeled data.

# MR-PHE: Multi-Resolution Prompt-guided Hybrid Embedding

A zero-shot learning framework for histopathology image classification using pretrained vision-language models and prompt engineering.

## 🔬 Purpose
MR-PHE classifies histopathology images **without any fine-tuning or labeled training data** by:
- Extracting multi-scale patches.
- Using frozen encoders from the CONCH model.
- Constructing hybrid embeddings.
- Comparing them with weighted text embeddings generated from domain-specific prompts.

## 🧱 Architecture

### 1. Preprocessing
- Resize image at scales `S = {0.25, 0.5, 0.75}`
- Extract `n = 5` random patches per scale using crop function `C`
- Combine all patches and the original image into set `P`

### 2. Embedding Generation
- Compute normalized image embeddings `e_p` for each `x_p ∈ P`
- Compute global embedding from original image
- Normalize all embeddings

### 3. Text Prompt Engineering
For each class:
- Generate prompt set `T_c` from:
  - Synonyms (e.g., malignant → “aggressive”, “cancerous”)
  - Templates (e.g., "An H&E stained image of {class}")
  - Clinical descriptors (GPT-4 generated, curated)
- Encode prompts using text encoder `g`
- Normalize embeddings

### 4. Prompt Evaluation & Selection
- Evaluate each prompt's accuracy on validation set
- Select top-K prompts per class (K = 30)

### 5. Class Embedding Construction
- Use similarity with reference label embedding to weight each prompt
- Final class embedding is weighted average of top-K normalized prompt embeddings

### 6. Hybrid Embedding Formation
- Compute patch attention weights based on max class similarity
- Compute weighted patch embedding: `e_patch`
- Combine with global: `h = α * e_global + (1 - α) * e_patch` (α = 0.5)
- Normalize hybrid embedding

### 7. Classification
- Compute cosine similarity: `S = h^T * t_c`
- Apply temperature-scaled softmax (τ = 1.5)
- Predict class with highest probability

## 🧪 Datasets
Evaluated on 6 datasets:
- BRACS, CRC100K, EBHI, HE-GHI-DS, WSSS4LUAD, In-House Breast Dataset

## 📈 Performance
| Dataset       | F1 Score ↑ | Accuracy ↑ |
|---------------|------------|------------|
| BRACS (2Cls)  | 90.36%     | 95.80%     |
| CRC100K       | 76.95%     | 80.22%     |
| EBHI          | 94.46%     | 94.47%     |
| HE-GHI-DS     | 95.16%     | 96.71%     |
| WSSS4LUAD     | 89.56%     | 88.96%     |
| In-House      | 91.47%     | 93.26%     |

## ⚙️ Dependencies

- Python 3.12.3
- PyTorch 2.3.0
- CONCH Model encoders (image + text)
- NVIDIA DGX-A100 GPU (80GB VRAM for experiments)

## 🧪 Hyperparameters

| Parameter          | Value     |
|--------------------|-----------|
| α (Hybrid Weight)  | 0.5       |
| τ (Temp. Scale)    | 1.5       |
| β (Text Softmax)   | 2.0       |
| K (Top Prompts)    | 30        |
| Scales             | {0.25, 0.5, 0.75} |
| Patches per Scale  | 5         |

## 🧠 Citation
If you use MR-PHE, cite:
> Rahaman et al., *Leveraging Vision-Language Embeddings for Zero-Shot Learning in Histopathology Images*, IEEE JBHI, 2025. [DOI: 10.1109/JBHI.2025.3584802]

## 📎 Source Code
📁 GitHub: [Mamunur-20/MR-PHE](https://github.com/Mamunur-20/MR-PHE)

## ⚖️ License
Licensed under Creative Commons Attribution 4.0 International License (CC BY 4.0).
