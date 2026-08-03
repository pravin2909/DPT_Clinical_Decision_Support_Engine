# DPT Clinical Decision Support Engine — Technical Report

## Diagnostic, Prognostic & Therapeutic (DPT) System

---

## 1. Datasets

### 1.1 Eye Disease Dataset

| Property | Detail |
|---|---|
| **Source** | Kaggle: `gunavenkatdoddi/eye-diseases-classification` |
| **Domain** | Ophthalmology (Retinal Scans) |
| **Total Images** | ~4,217 |
| **Format** | ImageFolder (class-wise subdirectories) |
| **Image Type** | Fundus / Retinal scan photographs |

| Class | Description |
|---|---|
| `normal` | Healthy retina with no pathological findings |
| `cataract` | Clouding of the eye's natural lens |
| `diabetic_retinopathy` | Retinal damage caused by diabetes |
| `glaucoma` | Optic nerve damage due to elevated intraocular pressure |

### 1.2 Dental Disease Dataset

| Property | Detail |
|---|---|
| **Source** | Kaggle: `salmansajid05/oral-diseases` |
| **Domain** | Dentistry (Oral Photographs) |
| **Total Images** | ~15,373 |
| **Total Size** | ~246 MB |
| **Format** | ImageFolder with annotated images |
| **Image Type** | Intra-oral clinical photographs |
| **Augmentation** | Pre-augmented by dataset provider (rotation, flipping, scaling, noise) |

| Class | Description |
|---|---|
| `Calculus` | Dental calculus / tartar buildup on tooth surfaces |
| `Caries` | Tooth decay and cavities |
| `Gingivitis` | Inflammation or infection of the gums |
| `Hypodontia` | Congenital absence of one or more teeth |
| `Mouth Ulcer` | Oral ulcers / canker sores |
| `Tooth Discoloration` | Staining or discoloration of teeth |
| `Data caries` | Additional caries variant from dataset |

### 1.3 Skin Disease Dataset

| Property | Detail |
|---|---|
| **Source** | Kaggle: `ismailpromus/skin-diseases-image-dataset` |
| **Domain** | Dermatology (Skin Lesion Images) |
| **Total Images** | 27,153 |
| **Format** | ImageFolder, split into train/test by provider |
| **Image Type** | Clinical and dermoscopic skin lesion photographs |

| Class | Images | Description |
|---|---|---|
| Eczema | 1,677 | Inflammatory skin condition with itchy, red patches |
| Melanoma | 3,140 | Malignant skin tumor from melanocytes |
| Atopic Dermatitis | 1,257 | Chronic inflammatory skin disease |
| Basal Cell Carcinoma (BCC) | 3,323 | Most common type of skin cancer |
| Melanocytic Nevi (NV) | 7,970 | Benign moles composed of melanocytes |
| Benign Keratosis (BKL) | 2,079 | Non-cancerous keratotic skin growths |
| Psoriasis / Lichen Planus | 2,055 | Autoimmune inflammatory skin conditions |
| Seborrheic Keratoses | 1,847 | Benign warty skin growths |
| Tinea / Ringworm / Candidiasis | 1,702 | Fungal skin infections |
| Warts / Molluscum | 2,103 | Viral skin infections |

> **Note:** The dataset exhibits class imbalance — Melanocytic Nevi (7,970) has ~6.3× more images than Atopic Dermatitis (1,257). This impacts the classifier's per-class performance.

### 1.4 MobileNet Router Custom Dataset

| Property | Detail |
|---|---|
| **Source** | Combined from all 3 datasets above |
| **Purpose** | Train the gateway image router (3-class classification) |
| **Total Images** | ~12,000 (up to 4,000 per domain) |
| **Creation Method** | Random sampling with uniform cap per class |
| **Format** | ImageFolder with 3 folders: `eye/`, `dental/`, `skin/` |

| Class | Source Dataset | Max Sampled |
|---|---|---|
| `eye` | Eye Diseases Classification | 4,000 |
| `dental` | Oral Diseases | 4,000 |
| `skin` | Skin Diseases Image Dataset | 4,000 |

---

## 2. Model Architectures

### 2.1 Vision Transformer (ViT) — Custom Architecture

All three domain-specific classifiers (eye, dental, skin) use the **same custom ViT architecture** with two key innovations: **Shifted Patch Tokenization (SPT)** and **Locality Self-Attention (LSA)**.

#### Architecture Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `img_size` | 72 × 72 | Input image resolution (RGB) |
| `patch_size` | 6 × 6 | Size of each image patch |
| `grid_size` | 12 × 12 | Number of patches per dimension (72/6) |
| `num_patches` | 144 | Total patches per image (12²) |
| `embed_dim` | 64 | Embedding dimension for each token |
| `depth` | 4 | Number of Transformer blocks |
| `num_heads` | 4 | Attention heads per block |
| `head_dim` | 16 | Dimension per head (64/4) |
| `mlp_dim` | 128 | Hidden dimension of MLP (2× embed_dim) |
| `dropout` | 0.1 | Dropout rate throughout the network |

#### Total Parameters: ~21.16 million

#### Architecture Diagram

```
Input Image (3 × 72 × 72)
        │
        ▼
┌─────────────────────────────────────┐
│  SHIFTED PATCH TOKENIZATION (SPT)   │
│                                     │
│  1. Pad image by shift_size (3px)   │
│  2. Create 5 versions:              │
│     - Original                      │
│     - Shift Left                    │
│     - Shift Right                   │
│     - Shift Up                      │
│     - Shift Down                    │
│  3. Concatenate → (15 × 72 × 72)   │
│  4. Unfold into 144 patches         │
│  5. Linear projection:              │
│     (15 × 6 × 6 = 540) → 64        │
│                                     │
│  Output: (B, 144, 64)              │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  POSITIONAL ENCODING                │
│                                     │
│  Learnable position embeddings      │
│  Shape: (1, 144, 64)               │
│  Initialized: truncated normal      │
│  (std=0.02)                         │
│                                     │
│  tokens = tokens + pos_embed        │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  TRANSFORMER BLOCK × 4             │
│  ┌─────────────────────────────┐    │
│  │ LayerNorm                   │    │
│  │     ↓                       │    │
│  │ Multi-Head LSA (4 heads)    │    │
│  │   - Q, K, V projection     │    │
│  │   - Scaled dot-product attn │    │
│  │   - Learnable temperature τ │    │
│  │   - Mean subtraction (LSA)  │    │
│  │   - Softmax + Dropout       │    │
│  │     ↓                       │    │
│  │ Residual connection + Drop  │    │
│  │     ↓                       │    │
│  │ LayerNorm                   │    │
│  │     ↓                       │    │
│  │ MLP: 64→128 (GELU) →64     │    │
│  │     ↓                       │    │
│  │ Residual connection         │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  CLASSIFICATION HEAD                │
│                                     │
│  LayerNorm                          │
│     ↓                               │
│  Flatten (144 × 64 = 9,216)        │
│     ↓                               │
│  Dropout(0.1)                       │
│     ↓                               │
│  Linear(9216 → 2048) + ReLU        │
│     ↓                               │
│  Dropout(0.1)                       │
│     ↓                               │
│  Linear(2048 → 1024) + ReLU        │
│     ↓                               │
│  Dropout(0.1)                       │
│     ↓                               │
│  Linear(1024 → num_classes)         │
│     ↓                               │
│  Output: class logits               │
└─────────────────────────────────────┘
```

#### 2.1.1 Shifted Patch Tokenization (SPT) — Detail

Standard ViT patch tokenization chops the image into non-overlapping patches, losing information at patch boundaries. SPT addresses this by:

1. **Padding** the image by `patch_size // 2` pixels on all sides
2. **Extracting 5 versions** of the image by cropping from different positions (original, left-shifted, right-shifted, up-shifted, down-shifted)
3. **Concatenating** along the channel dimension: 5 × 3 channels = **15 channels**
4. **Unfolding** into patches and projecting to the embedding dimension

This gives each patch **contextual awareness** of its neighboring pixels without increasing the number of patches.

**Input to projection layer:** 5 × 3 (channels) × 6 × 6 (patch) = **540 dimensions**

**Output of projection layer:** **64 dimensions** (embed_dim)

#### 2.1.2 Locality Self-Attention (LSA) — Detail

Standard self-attention treats all token relationships equally. LSA improves locality sensitivity:

1. **Learnable Temperature (τ):** Each attention head has a learnable scalar temperature parameter that scales the attention logits. This allows each head to independently control the sharpness of its attention distribution.

2. **Mean Subtraction:** Before applying softmax, the mean of attention scores is subtracted:
   ```
   attn = (Q × K^T) × scale × τ
   attn = attn - mean(attn, dim=-1)   ← LSA-specific
   attn = softmax(attn)
   ```
   This makes the attention distribution more peaked, encouraging the model to focus on **local, relevant patterns** rather than distributing attention uniformly — critical for small datasets where the model might otherwise overfit to noise.

**Temperature parameter shape:** `(num_heads, 1, 1)` — one scalar per head.

---

### 2.2 MobileNet V2 — Image Router

| Property | Detail |
|---|---|
| **Base Architecture** | MobileNet V2 (Sandler et al., 2018) |
| **Pretrained On** | ImageNet-1K (1.28M images, 1000 classes) |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Total Parameters** | 2,881,283 |
| **Trainable (Phase 1)** | 657,411 (classifier head only) |
| **Trainable (Phase 2)** | 2,881,283 (all layers) |
| **Output** | 3 classes: `eye`, `dental`, `skin` |

#### Architecture Summary

```
Input (3 × 224 × 224)
        │
        ▼
┌─────────────────────────────────────┐
│  MOBILENET V2 BACKBONE              │
│  (pretrained on ImageNet)           │
│                                     │
│  - Initial Conv2d (3→32, stride=2)  │
│  - 17 Inverted Residual Blocks      │
│    with depthwise separable convs   │
│    and linear bottleneck design     │
│  - Final Conv2d (320→1280)          │
│  - Global Average Pooling           │
│                                     │
│  Output: 1280-dim feature vector    │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  CUSTOM CLASSIFIER HEAD             │
│                                     │
│  Dropout(0.3)                       │
│     ↓                               │
│  Linear(1280 → 512) + ReLU         │
│     ↓                               │
│  Dropout(0.2)                       │
│     ↓                               │
│  Linear(512 → 3)                    │
│     ↓                               │
│  Output: 3 class logits             │
│  (eye / dental / skin)              │
└─────────────────────────────────────┘
```

#### Key Design: Inverted Residual Block

MobileNet V2's core innovation is the **inverted residual block** with depthwise separable convolutions:

```
Input (narrow, e.g. 16 channels)
   ↓
Pointwise Conv (expand: 16 → 96)     ← "Expansion"
   ↓
Depthwise Conv (96 → 96, 3×3)        ← "Per-channel filtering"
   ↓
Pointwise Conv (compress: 96 → 24)   ← "Projection"
   ↓
Residual Add (if input/output same)
```

This design is **~10× fewer operations** than standard convolutions, making it ideal for real-time inference on low-compute devices.

#### Transfer Learning Strategy

| Phase | Epochs | What's Trained | Learning Rate | Params |
|---|---|---|---|---|
| **Phase 1** (Frozen) | 1–3 | Classifier head only | 1e-3 | 657K |
| **Phase 2** (Unfrozen) | 4–15 | All layers (differential LR) | Backbone: 1e-4, Head: 5e-4 | 2.88M |

**Rationale:** The ImageNet-pretrained backbone already knows how to extract visual features (edges, textures, shapes). Phase 1 trains the new classifier head without disrupting these features. Phase 2 gently fine-tunes the entire network to adapt features for medical image domain classification.

---

## 3. Training Configuration

### 3.1 Shared Training Settings

| Parameter | ViT Models | MobileNet Router |
|---|---|---|
| **Optimizer** | AdamW | AdamW |
| **Base Learning Rate** | 3e-4 | 1e-3 |
| **Minimum LR** | 1e-6 | 1e-6 |
| **Weight Decay** | 1e-2 | 1e-4 |
| **LR Schedule** | Warmup (3 epochs) + Cosine Annealing | Warmup (2 epochs) + Cosine Annealing |
| **Label Smoothing** | 0.1 | 0.05 |
| **Gradient Clipping** | max_norm = 1.0 | max_norm = 1.0 |
| **Mixed Precision** | FP16 (AMP) on GPU | FP16 (AMP) on GPU |
| **Batch Size** | 32 | 64 |
| **Early Stopping** | Patience = 8 | Patience = 6 |
| **Validation Split** | 80% train / 20% val | 80% train / 20% val |

### 3.2 Data Augmentation

#### ViT Models (Training Transforms)

```python
transforms.Compose([
    Resize((84, 84)),                              # Slightly larger than target
    RandomResizedCrop(72, scale=(0.7-0.75, 1.0)),  # Random crop to 72×72
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.2-0.3),
    RandomRotation(20-25°),
    ColorJitter(brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.05-0.1),
    RandomGrayscale(p=0.05),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225]),               # ImageNet normalization
    RandomErasing(p=0.2-0.25, scale=(0.02, 0.15-0.2)),
])
```

#### ViT Models (Validation Transforms)

```python
transforms.Compose([
    Resize((72, 72)),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225]),
])
```

#### MobileNet Router (Training Transforms)

```python
transforms.Compose([
    Resize((256, 256)),
    RandomResizedCrop(224, scale=(0.7, 1.0)),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.2),
    RandomRotation(20°),
    ColorJitter(brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.05),
    RandomGrayscale(p=0.05),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225]),
    RandomErasing(p=0.15, scale=(0.02, 0.15)),
])
```

### 3.3 Loss Function

**Cross-Entropy Loss with Label Smoothing:**

```
L = -(1 - ε) × log(p_target) - ε/(K-1) × Σ log(p_other)
```

Where:
- `ε` = smoothing factor (0.1 for ViTs, 0.05 for MobileNet)
- `K` = number of classes
- Instead of hard labels [0, 0, 1, 0], uses soft labels [0.033, 0.033, 0.9, 0.033]

**Purpose:** Prevents the model from becoming overconfident on training samples, improving generalization to unseen medical images.

---

## 4. Performance Metrics

### 4.1 MobileNet V2 Router

| Metric | Value |
|---|---|
| **Best Validation Accuracy** | **99.92%** |
| **Training Epochs** | ~9 (early stopping triggered) |
| **Training Platform** | Kaggle (NVIDIA Tesla T4, 15.6 GB VRAM) |
| **Convergence** | Phase 1 reached ~99% in 3 epochs; Phase 2 refined to 99.92% |

**Per-Class Routing (during inference test):**

| Input Type | Predicted | Confidence |
|---|---|---|
| Skin image (Melanoma.jpg) | `skin` | 97.91% |
| Retinal scan | `eye` | ~98%+ |
| Dental photograph | `dental` | ~99%+ |

> The router achieves near-perfect classification because the three image domains (retinal scans, dental photos, skin lesions) are visually very distinct.

### 4.2 Skin Disease ViT

| Metric | Value |
|---|---|
| **Best Validation Accuracy** | **67.55%** |
| **Total Training Epochs** | 30 (completed full run) |
| **Training Platform** | Kaggle (NVIDIA Tesla T4) |
| **Total Parameters** | 21,161,946 |
| **Random Baseline** | 10% (1/10 classes) |
| **Improvement over random** | ~6.75× |

**Training Progression:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 1.7496 | 41.45% | 1.5925 | 48.42% |
| 5 | 1.4919 | 53.01% | 1.5181 | 55.06% |
| 10 | 1.3979 | 57.54% | 1.3635 | 61.55% |
| 15 | 1.3324 | 60.85% | 1.3198 | 62.73% |
| 20 | 1.2640 | 64.33% | 1.2772 | 66.02% |
| 25 | 1.2095 | 66.89% | 1.2239 | 67.00% |
| 30 | 1.1915 | 67.84% | 1.2122 | **67.55%** |

**Factors Affecting Performance:**
1. **Class imbalance:** Melanocytic Nevi has 7,970 images vs. Atopic Dermatitis with 1,257 — a 6.3:1 ratio
2. **Small input size:** 72×72 resolution limits fine-grained feature extraction for skin lesions
3. **Training from scratch:** No pretrained weights — the model learns all features from medical images only
4. **Inter-class similarity:** Several skin conditions (BCC, Seborrheic Keratoses, Benign Keratosis) share visual characteristics

### 4.3 Eye Disease ViT

| Metric | Value |
|---|---|
| **Classes** | 4 (normal, cataract, diabetic retinopathy, glaucoma) |
| **Total Parameters** | ~21M |
| **Training Configuration** | Same ViT architecture, 20 epochs |
| **Training Dataset** | ~4,217 images |

### 4.4 Dental Disease ViT

| Metric | Value |
|---|---|
| **Classes** | 7 |
| **Total Parameters** | ~21M |
| **Training Configuration** | Same ViT architecture, 30 epochs |
| **Training Dataset** | ~15,373 images |

### 4.5 Model Size Comparison

| Model | Parameters | File Size | Inference Input |
|---|---|---|---|
| MobileNet V2 Router | 2.88M | ~34 MB | 224 × 224 |
| Eye ViT | ~21.16M | ~254 MB | 72 × 72 |
| Dental ViT | ~21.16M | ~254 MB | 72 × 72 |
| Skin ViT | ~21.16M | ~254 MB | 72 × 72 |
| **Total (all 4)** | **~66.36M** | **~796 MB** | — |
| Gemma 3 4B (LLM) | ~4B | ~3.3 GB | Text tokens |

---

## 5. Reasoning Model

### 5.1 Gemma 3 4B

| Property | Detail |
|---|---|
| **Model** | Gemma 3 4B |
| **Provider** | Google DeepMind |
| **Deployment** | Local via Ollama |
| **Quantization** | Default (Q4_K_M GGUF) |
| **Size on Disk** | ~3.3 GB |
| **API Endpoint** | `http://localhost:11434/api/generate` |
| **Context Window** | 8,192 tokens |
| **Role in Pipeline** | Generate structured DPT clinical reports from classified disease names or symptom descriptions |

### 5.2 Inference Settings

| Parameter | Value | Rationale |
|---|---|---|
| `temperature` | 0.7 | Balanced between creativity and factual accuracy |
| `top_p` | 0.9 | Nucleus sampling for coherent medical text |
| `num_predict` | 2048 | Maximum tokens for a complete DPT report |

---

## 6. Inference Pipeline Timing

| Stage | Approximate Time (CPU) | Approximate Time (GPU) |
|---|---|---|
| MobileNet Routing | ~50ms | ~10ms |
| ViT Classification | ~200ms | ~30ms |
| Gemma 3 Report Generation | 25–35 seconds | 10–15 seconds |
| **Total (Image Pipeline)** | **~30 seconds** | **~12 seconds** |
| **Total (Text Pipeline)** | **~25–35 seconds** | **~10–15 seconds** |

> Report generation by the LLM dominates total inference time. The vision models are negligible in comparison.

---

## 7. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Deep Learning Framework | PyTorch | ≥ 2.0 |
| Image Processing | torchvision, Pillow | Latest |
| API Framework | FastAPI | 0.115.0 |
| ASGI Server | Uvicorn | 0.30.0 |
| LLM Serving | Ollama | Latest |
| LLM Communication | httpx (async HTTP) | 0.27.0 |
| Speech Recognition | SpeechRecognition | ≥ 3.10.0 |
| Language | Python | 3.10+ |
| Training Platform | Kaggle Notebooks / Google Colab | T4 GPU |

---

## 8. File Structure

```
DPT clinical decision support engine/
│
├── Models/                              ← Trained model weights
│   ├── mobilenet_router.pth               (34.9 MB)
│   ├── vit_eye_disease.pth                (253.9 MB)
│   ├── vit_dental_disease.pth             (254.0 MB)
│   └── vit_skin_disease.pth               (254.0 MB)
│
├── app/                                 ← Core pipeline modules
│   ├── __init__.py
│   ├── config.py                          Configuration & prompts
│   ├── model_loader.py                    ViT architecture + loaders
│   ├── pipeline.py                        Orchestrator (Router → ViT → LLM)
│   ├── reasoning.py                       Ollama / Gemma 3 integration
│   └── speech.py                          Speech-to-Text conversion
│
├── main.py                              ← FastAPI server entry point
├── requirements.txt                     ← Python dependencies
├── project_description.md               ← Project overview
├── technical_report.md                  ← This document
│
├── vit_eye_disease.py                   ← Training script (Eye ViT)
├── vit_dental_disease.py                ← Training script (Dental ViT)
├── vit_skin_disease.py                  ← Training script (Skin ViT)
├── mobilenet_router.py                  ← Training script (MobileNet Router)
└── mobilenet_inference_cell.py          ← Kaggle inference testing cell
```

---

## 9. Checkpoint Format

Each `.pth` file contains a dictionary with the following keys:

### ViT Checkpoints

```python
{
    'epoch': int,                    # Best epoch number
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'best_val_acc': float,           # Best validation accuracy
    'classes': list,                 # Class names (folder order)
    'class_to_idx': dict,            # {"class_name": index}
    'idx_to_class': dict,            # {index: "class_name"}
    'num_classes': int,              # Number of output classes
    'model_config': {
        'img_size': 72,
        'patch_size': 6,
        'embed_dim': 64,
        'depth': 4,
        'num_heads': 4,
        'mlp_dim': 128,
        'dropout': 0.1,
    },
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'domain': str,                   # "eye" / "dental" / "skin"
}
```

### MobileNet Checkpoint

```python
{
    # Same as above, plus:
    'model_config': {
        'architecture': 'mobilenet_v2',
        'img_size': 224,
        'pretrained': True,
        'classifier': [1280, 512, 3],
    },
    'routing_map': {
        'eye': {'vit_model': 'vit_eye_disease.pth', 'num_classes': 4, 'classes': [...]},
        'dental': {'vit_model': 'vit_dental_disease.pth', 'num_classes': 6, 'classes': [...]},
        'skin': {'vit_model': 'vit_skin_disease.pth', 'num_classes': 10, 'classes': [...]},
    },
    'domain': 'router',
}
```

---

## 10. API Endpoints

| Method | Endpoint | Input | Output |
|---|---|---|---|
| `GET` | `/health` | — | System status, Ollama status, device |
| `GET` | `/models/status` | — | Loaded models, class lists |
| `POST` | `/analyze/image` | Image file (multipart) | Routing + Disease + DPT Report |
| `POST` | `/analyze/text` | Symptoms text (form) | DPT Report |
| `POST` | `/analyze/audio` | Audio file (multipart) | Transcription + DPT Report |
| `POST` | `/analyze/image/stream` | Image file (multipart) | SSE streaming DPT Report |

---

## 11. Known Limitations

1. **Skin ViT accuracy (67.55%):** Moderate performance due to class imbalance, small input size (72×72), and training from scratch. Per-class accuracy varies significantly.

2. **Input resolution (72×72 for ViTs):** Medical images contain fine-grained details that are lost at this resolution. Larger input sizes (128×128 or 224×224) would significantly improve accuracy but require more compute.

3. **No pretrained ViT weights:** Training from scratch on small medical datasets limits feature learning. Using pretrained ViT (e.g., from ImageNet or medical pretraining) would dramatically improve performance.

4. **LLM inference speed:** Gemma 3 4B on CPU takes 25–35 seconds per report. GPU inference or a smaller model would reduce this.

5. **Clean label mapping:** The skin dataset has messy folder names (e.g., "1. Eczema 1677"). The pipeline maps these to clean names, but edge cases may exist.

---
