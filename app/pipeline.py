"""
Core Pipeline Orchestrator.
Loads all models on startup and routes inputs through the correct pipeline.
"""

import torch
import torch.nn.functional as F
from PIL import Image

from app.config import (
    MOBILENET_PATH, VIT_CONFIGS,
    DPT_IMAGE_PROMPT_TEMPLATE, DPT_TEXT_PROMPT_TEMPLATE,
)
from app.model_loader import (
    load_mobilenet_router, load_vit_classifier,
    get_mobilenet_transform, get_vit_transform,
    load_image_from_bytes,
)
from app.reasoning import generate_dpt_report, generate_dpt_report_stream, check_ollama_status
from app.speech import transcribe_audio


class DPTPipeline:
    """
    Main pipeline that orchestrates:
      Image  → MobileNet Router → ViT Classifier → Gemma 3 → DPT Report
      Text   → Gemma 3 → DPT Report
      Audio  → STT → Gemma 3 → DPT Report
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.router = None
        self.router_idx_to_class = None
        self.vit_models = {}
        self.vit_classes = {}
        self.mobilenet_transform = get_mobilenet_transform()
        self.vit_transform = get_vit_transform()
        self._loaded = False

    def load_models(self):
        """Load all models into memory. Call once at startup."""
        print("=" * 55)
        print("  🚀 DPT Pipeline — Loading Models")
        print("=" * 55)
        print(f"  Device: {self.device}")

        # 1. Load MobileNet Router
        print("\n  📦 Loading MobileNet V2 Router...")
        self.router, self.router_idx_to_class, _ = load_mobilenet_router(
            MOBILENET_PATH, self.device
        )
        print(f"     ✓ Router loaded → classes: {list(self.router_idx_to_class.values())}")

        # 2. Load ViT Classifiers
        for domain, cfg in VIT_CONFIGS.items():
            print(f"\n  📦 Loading {domain.upper()} ViT ({cfg['num_classes']} classes)...")
            model, idx_to_class, classes = load_vit_classifier(
                cfg["model_path"], cfg["num_classes"], self.device
            )
            self.vit_models[domain] = model
            self.vit_classes[domain] = {
                "idx_to_class": idx_to_class,
                "classes": classes,
            }
            print(f"     ✓ Loaded → {classes}")

        self._loaded = True
        print("\n" + "=" * 55)
        print("  ✅ All models loaded successfully!")
        print("=" * 55 + "\n")

    def is_loaded(self) -> bool:
        return self._loaded

    # ─────────────────────────────────
    #  Image Pipeline
    # ─────────────────────────────────
    @torch.no_grad()
    def classify_image_domain(self, image: Image.Image) -> dict:
        """
        Step 1: Use MobileNet to classify image as eye/dental/skin.
        Returns: { domain, confidence, all_probs }
        """
        inp = self.mobilenet_transform(image).unsqueeze(0).to(self.device)
        logits = self.router(inp)
        probs = F.softmax(logits, dim=1)[0]

        pred_idx = probs.argmax().item()
        pred_domain = self.router_idx_to_class[pred_idx]
        pred_conf = probs[pred_idx].item()

        all_probs = {
            self.router_idx_to_class[i]: round(probs[i].item() * 100, 2)
            for i in range(len(probs))
        }

        return {
            "domain": pred_domain,
            "confidence": round(pred_conf * 100, 2),
            "all_probabilities": all_probs,
        }

    @torch.no_grad()
    def classify_disease(self, image: Image.Image, domain: str) -> dict:
        """
        Step 2: Use domain-specific ViT to classify the disease.
        Returns: { disease, confidence, all_probs }
        """
        if domain not in self.vit_models:
            return {"error": f"Unknown domain: {domain}"}

        model = self.vit_models[domain]
        vit_info = self.vit_classes[domain]
        idx_to_class = vit_info["idx_to_class"]

        inp = self.vit_transform(image).unsqueeze(0).to(self.device)
        logits = model(inp)
        probs = F.softmax(logits, dim=1)[0]

        pred_idx = probs.argmax().item()
        pred_class = idx_to_class[pred_idx]
        pred_conf = probs[pred_idx].item()

        all_probs = {
            idx_to_class[i]: round(probs[i].item() * 100, 2)
            for i in range(len(probs))
        }

        # Clean up the class name
        clean_labels = VIT_CONFIGS[domain].get("clean_labels", {})
        clean_name = self._find_clean_label(pred_class, clean_labels)

        return {
            "disease": clean_name,
            "raw_class": pred_class,
            "confidence": round(pred_conf * 100, 2),
            "all_probabilities": all_probs,
        }

    def _find_clean_label(self, raw_class: str, clean_labels: dict) -> str:
        """Map raw folder-name class to a clean medical label."""
        # Try exact match
        if raw_class in clean_labels:
            return clean_labels[raw_class]

        # Try partial match (folder names from dataset can be messy)
        raw_lower = raw_class.lower()
        for key, value in clean_labels.items():
            if key.lower() in raw_lower or raw_lower in key.lower():
                return value

        # Fallback: clean up the raw name
        return raw_class.replace("_", " ").title()

    async def process_image(self, image_bytes: bytes) -> dict:
        """
        Full image pipeline:
          Image → Router → ViT → Gemma 3 → DPT Report
        """
        image = load_image_from_bytes(image_bytes)

        # Step 1: Route
        routing = self.classify_image_domain(image)
        domain = routing["domain"]

        # Step 2: Classify disease
        classification = self.classify_disease(image, domain)

        if "error" in classification:
            return {"error": classification["error"]}

        # Step 3: Generate DPT report
        domain_full_names = {
            "eye": "Ophthalmology (Eye)",
            "dental": "Dentistry (Oral Health)",
            "skin": "Dermatology (Skin)",
        }

        prompt = DPT_IMAGE_PROMPT_TEMPLATE.format(
            domain=domain,
            disease=classification["disease"],
            confidence=classification["confidence"],
            domain_full=domain_full_names.get(domain, domain),
        )

        report = await generate_dpt_report(prompt)

        return {
            "input_type": "image",
            "routing": routing,
            "classification": classification,
            "report": report,
        }

    # ─────────────────────────────────
    #  Text Pipeline
    # ─────────────────────────────────
    async def process_text(self, symptoms: str) -> dict:
        """
        Text pipeline:
          Symptoms text → Gemma 3 → DPT Report
        """
        prompt = DPT_TEXT_PROMPT_TEMPLATE.format(symptoms=symptoms)
        report = await generate_dpt_report(prompt)

        return {
            "input_type": "text",
            "symptoms": symptoms,
            "report": report,
        }

    # ─────────────────────────────────
    #  Audio Pipeline
    # ─────────────────────────────────
    async def process_audio(self, audio_bytes: bytes, filename: str) -> dict:
        """
        Audio pipeline:
          Audio → STT → Text → Gemma 3 → DPT Report
        """
        # Step 1: Transcribe
        transcription = transcribe_audio(audio_bytes, filename)

        if not transcription["success"]:
            return {
                "input_type": "audio",
                "error": transcription["error"],
            }

        symptoms = transcription["text"]

        # Step 2: Process as text
        result = await self.process_text(symptoms)
        result["input_type"] = "audio"
        result["transcription"] = symptoms

        return result


# Global pipeline instance
pipeline = DPTPipeline()
