"""
Configuration for the DPT Clinical Decision Support Engine.
All paths, model configs, and constants in one place.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional; env vars can still be set by the shell.
    pass

# ──────────────────────────────────────
# Paths
# ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# Model file paths
MOBILENET_PATH = os.path.join(MODELS_DIR, "mobilenet_router.pth")
EYE_VIT_PATH = os.path.join(MODELS_DIR, "vit_eye_disease.pth")
DENTAL_VIT_PATH = os.path.join(MODELS_DIR, "vit_dental_disease.pth")
SKIN_VIT_PATH = os.path.join(MODELS_DIR, "vit_skin_disease.pth")

# Upload directory for temporary files
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ──────────────────────────────────────
# MobileNet Router Config
# ──────────────────────────────────────
MOBILENET_IMG_SIZE = 224
MOBILENET_CLASSES = ["dental", "eye", "skin"]

# ──────────────────────────────────────
# ViT Config (shared across all 3 ViTs)
# ──────────────────────────────────────
VIT_IMG_SIZE = 72
VIT_PATCH_SIZE = 6
VIT_EMBED_DIM = 64
VIT_DEPTH = 4
VIT_NUM_HEADS = 4
VIT_MLP_DIM = 128
VIT_DROPOUT = 0.1

# ──────────────────────────────────────
# Domain-specific ViT Configs
# ──────────────────────────────────────
VIT_CONFIGS = {
    "eye": {
        "model_path": EYE_VIT_PATH,
        "num_classes": 4,
        "clean_labels": {
            "cataract": "Cataract",
            "diabetic_retinopathy": "Diabetic Retinopathy",
            "glaucoma": "Glaucoma",
            "normal": "Normal (Healthy Eye)",
        },
    },
    "dental": {
        "model_path": DENTAL_VIT_PATH,
        "num_classes": 7,
        "clean_labels": {
            "Calculus": "Dental Calculus (Tartar)",
            "Caries": "Dental Caries (Tooth Decay)",
            "Gingivitis": "Gingivitis",
            "Hypodontia": "Hypodontia (Missing Teeth)",
            "Mouth Ulcer": "Mouth Ulcer",
            "Tooth Discoloration": "Tooth Discoloration",
            "Ulcers": "Oral Ulcers",
        },
    },
    "skin": {
        "model_path": SKIN_VIT_PATH,
        "num_classes": 10,
        "clean_labels": {
            "Eczema": "Eczema",
            "Melanoma": "Melanoma",
            "Atopic Dermatitis": "Atopic Dermatitis",
            "Basal Cell Carcinoma (BCC)": "Basal Cell Carcinoma",
            "Melanocytic Nevi": "Melanocytic Nevi (Moles)",
            "Benign Keratosis-like Lesions": "Benign Keratosis",
            "Psoriasis": "Psoriasis",
            "Seborrheic Keratoses": "Seborrheic Keratoses",
            "Tinea Ringworm Candidiasis": "Tinea / Ringworm / Candidiasis",
            "Warts Molluscum": "Warts / Molluscum",
        },
    },
}

# ──────────────────────────────────────
# Image Normalization (ImageNet standard)
# ──────────────────────────────────────
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────
# LLM Reasoning Backend (OpenAI-compatible: LM Studio, Ollama, vLLM, etc.)
# ──────────────────────────────────────
# LM Studio serves an OpenAI-compatible API at http://localhost:1234/v1
# Ollama also exposes one at http://localhost:11434/v1
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-e4b")
# LM Studio ignores the key but the OpenAI-style API requires one to be present.
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")

# Backwards-compatible alias (older code referenced OLLAMA_MODEL).
OLLAMA_MODEL = LLM_MODEL

# ──────────────────────────────────────
# DPT Report System Prompt
# ──────────────────────────────────────
DPT_SYSTEM_PROMPT = """You are a medical clinical decision support system that generates structured DPT (Diagnostic, Prognostic, Therapeutic) reports.

STRICT RULES:
- Start the report DIRECTLY with the report title. Do NOT begin with "Okay", "Sure", "Here's", or any conversational filler.
- Do NOT invent or hallucinate dates. Do NOT include any date field.
- Do NOT ask follow-up questions at the end. The report must be self-contained.
- Do NOT use phrases like "Do you want me to..." or "Let me know if...".
- Use professional medical language that is also understandable to patients.
- Be factual and evidence-based. Do not speculate beyond established medical knowledge.

OUTPUT FORMAT (follow this exactly):

# DPT Clinical Report: [Condition Name]

**Detected Condition:** [Name]
**Medical Domain:** [Ophthalmology / Dentistry / Dermatology]

> ⚠️ DISCLAIMER: This report is AI-generated and intended for informational purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 1. DIAGNOSIS
- Clinical definition of the condition
- Key visual/clinical characteristics
- Common symptoms patients experience
- Standard diagnostic methods and procedures
- Differential diagnoses to consider

## 2. PROGNOSIS
- Natural progression of the condition (untreated)
- Expected outcomes with timely treatment
- Severity classification (mild / moderate / severe)
- Risk factors that worsen the prognosis
- Potential complications if left untreated

## 3. THERAPY
- First-line treatment options
- Medications (with generic names)
- Surgical or procedural interventions if applicable
- Lifestyle modifications and home care
- Preventive measures to avoid recurrence
- Red flags that require immediate specialist referral

---
End the report with a brief clinical summary in 2-3 sentences. Nothing else after that."""

DPT_IMAGE_PROMPT_TEMPLATE = """An AI-powered medical image analysis system has classified a patient's uploaded image.

CLASSIFICATION RESULTS:
- Detected Condition: {disease}
- Confidence Score: {confidence:.1f}%
- Image Domain: {domain_full}

Generate a DPT clinical report for this condition. Follow the system format strictly."""

DPT_TEXT_PROMPT_TEMPLATE = """A patient has described the following symptoms:

"{symptoms}"

Analyze these symptoms, identify the most likely condition(s), and generate a DPT clinical report. Follow the system format strictly."""

