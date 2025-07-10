🔬 Research Topics Deep Dive

Comprehensive Notes on Diffusion Models, Screenshot Models, Domain Adaptation, Knowledge Distillation, and Multimodal Text-Image Models

Last updated: July 2025


---

📌 1. Diffusion Models

➤ What Are They?

Diffusion models are a type of generative model that learns to generate data by reversing a noise-adding process.

🔁 Process:

Forward Process: Add Gaussian noise to an image until it becomes pure noise.

Reverse Process: Learn to reverse the noise process to reconstruct the original image.


➤ Key Components

UNet Architecture: Core neural network that denoises images.

Noise Schedule: Determines how noise is added and removed.

Training Objective: Score matching or denoising objective.

Conditional Diffusion: Can be conditioned on text, labels, or other images.


➤ Types

DDPM: Denoising Diffusion Probabilistic Models.

Latent Diffusion: Compress images to a latent space to speed up training (used in Stable Diffusion).

Score-based models: Use gradients of data density.


➤ Applications

Text-to-image generation (Stable Diffusion, Imagen).

Audio synthesis.

3D shape generation.


➤ Key Papers

DDPM: https://arxiv.org/abs/2006.11239

Latent Diffusion (Stable Diffusion): https://arxiv.org/abs/2112.10752

Imagen: https://arxiv.org/abs/2205.11487


➤ Libraries

HuggingFace Diffusers: https://huggingface.co/docs/diffusers

CompVis Stable Diffusion

OpenAI Glide



---

📌 2. Screenshot Models

➤ What Are They?

Models trained to understand and process screenshots of apps, websites, or digital documents.

➤ Core Tasks

OCR (Optical Character Recognition): Extract text from screenshots.

UI Element Detection: Detect buttons, icons, etc.

Screen Parsing: Understand the hierarchy and structure of a screen.


➤ Techniques

OCR libraries: EasyOCR, PaddleOCR, Tesseract

Layout models: Donut, LayoutLM, Screen2Vec

Object Detection: YOLO, Faster R-CNN for UI components.


➤ Applications

App automation/testing.

Digital accessibility.

UI summarization.


➤ Example Models

LayoutLM: Pre-trained on document layouts.

Donut: End-to-end OCR-free document understanding.

Screen parsing models from Google AI / Facebook AI.



---

📌 3. Domain Adaptation

➤ What Is It?

Technique to adapt a model trained on a source domain to perform well on a target domain.

➤ Types of Domain Adaptation

Unsupervised: No labels in the target domain.

Supervised: Small amount of labeled target data.

Semi-supervised: Few labeled target samples + unlabeled.


➤ Common Techniques

Domain-Adversarial Training: Use a domain discriminator to align features (DANN).

Maximum Mean Discrepancy (MMD): Minimize feature distribution differences.

Contrastive Learning: Align representations across domains.


➤ Example Domains

Photos to sketches.

Real-world images to cartoons.

English to German NLP.


➤ Key Papers

DANN: https://arxiv.org/abs/1505.07818

SHOT: https://arxiv.org/abs/2006.07849


➤ Libraries

PyTorch DANN implementations.

TensorFlow Domain Adaptation Library.



---

📌 4. Knowledge Distillation

➤ What Is It?

A method to compress a large (teacher) model into a smaller (student) model by transferring knowledge.

➤ Why Use It?

Reduce model size for deployment.

Speed up inference.

Retain accuracy in a smaller model.


➤ Types of Distillation

Logit Distillation: Student mimics soft output probabilities.

Feature Distillation: Student mimics hidden feature representations.

Relational Distillation: Student matches the similarity between samples.


➤ Example Scenarios

BERT → DistilBERT, TinyBERT.

Vision Transformers → MobileNet.


➤ Key Paper

"Distilling the Knowledge in a Neural Network" by Hinton et al.: https://arxiv.org/abs/1503.02531


➤ Libraries

Huggingface Transformers

PyTorch Lightning Distillation callbacks.



---

📌 5. Multimodal Text-Image Models

➤ What Are They?

Models that learn to process and align text and image data in a joint embedding space.

➤ Core Tasks

Image Captioning: Generate captions for images.

Text-to-Image Generation: Generate images from text prompts.

Visual Question Answering (VQA): Answer questions about images.

Image Retrieval: Retrieve images based on text descriptions.


➤ Key Architectures

CLIP (OpenAI): Contrastive pretraining on 400M image-text pairs.

BLIP / BLIP-2: Unified pretraining + fine-tuning for vision-language tasks.

Flamingo (DeepMind): Few-shot vision-language model.

LLaVA: Vision-Language model fine-tuned on LLaMA.


➤ Techniques

Contrastive Learning: Align text and image features.

Masked Language/Image Modeling: Predict masked parts of inputs.

Cross-Attention: Fuse image and text features.


➤ Key Papers

CLIP: https://arxiv.org/abs/2103.00020

BLIP-2: https://arxiv.org/abs/2301.12597

LLaVA: https://arxiv.org/abs/2304.08485


➤ Libraries

OpenCLIP: https://github.com/mlfoundations/open_clip

Huggingface Transformers (VisionEncoderDecoderModel, etc.)



---

🚀 Suggested Learning Order

Priority	Topic	Tools to Practice

1	Diffusion Models	Diffusers, Stable Diffusion
2	Multimodal Text-Image Models	CLIP, BLIP, Huggingface Transformers
3	Knowledge Distillation	PyTorch, TinyBERT, DistilBERT
4	Domain Adaptation	DANN, MMD, SHOT
5	Screenshot Models (if relevant)	PaddleOCR, Donut, LayoutLM



---

🔍 Suggested Projects

Diffusion: Build a text-to-image generator on CIFAR-10 or CelebA.

Screenshot Models: Parse Android app screenshots to detect buttons and texts.

Domain Adaptation: Train on MNIST, adapt to MNIST-M.

Knowledge Distillation: Distill CLIP into a smaller ResNet-based model.

Multimodal: Build a product image search using CLIP embeddings.



---

🔗 Recommended Resources

Type	Resource

Course	Huggingface Diffusion Models Course
Tutorials	PyImageSearch, PapersWithCode, GitHub trending repos
Datasets	COCO, Flickr30k, Office-31, MNIST-M, Rico
Blogs	OpenAI, DeepMind, Huggingface, Google AI Blog



---

✅ Final Tips

Keep experimenting with small datasets first.

Reproduce at least 1-2 papers from scratch.

Read the original paper + blog + GitHub code for each model.

Stay updated via arXiv-sanity, PapersWithCode, Huggingface updates.



---

Feel free to contribute to this repo with new papers or implementation insights!

