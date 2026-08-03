"""
═══════════════════════════════════════════════════════════════
  DPT Clinical Decision Support Engine — FastAPI Server
  
  Endpoints:
    POST /analyze/image   → Upload medical image → DPT report
    POST /analyze/text    → Submit symptoms text → DPT report
    POST /analyze/audio   → Upload audio/speech  → DPT report
    GET  /health          → System health check
    GET  /models/status   → Model loading status
═══════════════════════════════════════════════════════════════
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from contextlib import asynccontextmanager
import json
import time
import os

from app.pipeline import pipeline
from app.reasoning import check_ollama_status
from app.config import OLLAMA_MODEL


# ──────────────────────────────────────
# Lifespan: Load models on startup
# ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models when the server starts."""
    pipeline.load_models()
    yield
    print("🛑 DPT Engine shutting down.")


# ──────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────
app = FastAPI(
    title="DPT Clinical Decision Support Engine",
    description=(
        "AI-powered Diagnostic, Prognostic & Therapeutic clinical report generator. "
        "Accepts medical images, symptom text, or audio input."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend (Google Stitch or any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────
# UI Serving Endpoint
# ──────────────────────────────────────
@app.get("/")
async def serve_ui():
    """Serve the root UI."""
    html_path = "index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>UI mapping pending...</h1>")


# ──────────────────────────────────────
# Health & Status Endpoints
# ──────────────────────────────────────
@app.get("/health")
async def health_check():
    """System health check."""
    ollama_ok = await check_ollama_status()
    return {
        "status": "healthy" if pipeline.is_loaded() and ollama_ok else "degraded",
        "models_loaded": pipeline.is_loaded(),
        "ollama_available": ollama_ok,
        "ollama_model": OLLAMA_MODEL,
        "device": str(pipeline.device),
    }


@app.get("/models/status")
async def models_status():
    """Detailed model loading status."""
    return {
        "router": {
            "loaded": pipeline.router is not None,
            "classes": list(pipeline.router_idx_to_class.values()) if pipeline.router_idx_to_class else [],
        },
        "vit_models": {
            domain: {
                "loaded": domain in pipeline.vit_models,
                "classes": info.get("classes", []),
            }
            for domain, info in pipeline.vit_classes.items()
        } if pipeline.vit_classes else {},
    }


# ──────────────────────────────────────
# Image Analysis Endpoint
# ──────────────────────────────────────
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload a medical image (retinal scan, dental photo, or skin image).
    Returns:
      - Routing result (eye/dental/skin)
      - Disease classification
      - DPT clinical report
    """
    if not pipeline.is_loaded():
        raise HTTPException(status_code=503, detail="Models are still loading. Please wait.")

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Accepted: JPG, PNG, WebP, BMP",
        )

    start = time.time()
    image_bytes = await file.read()

    result = await pipeline.process_image(image_bytes)
    result["processing_time_seconds"] = round(time.time() - start, 2)

    return JSONResponse(content=result)


# ──────────────────────────────────────
# Text Analysis Endpoint
# ──────────────────────────────────────
@app.post("/analyze/text")
async def analyze_text(symptoms: str = Form(...)):
    """
    Submit symptom description text.
    Returns a DPT clinical report based on the described symptoms.
    """
    if not symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms text cannot be empty.")

    start = time.time()
    result = await pipeline.process_text(symptoms.strip())
    result["processing_time_seconds"] = round(time.time() - start, 2)

    return JSONResponse(content=result)


@app.post("/analyze/text/stream")
async def analyze_text_stream(symptoms: str = Form(...)):
    """
    Stream DPT report token-by-token for text symptoms.
    """
    if not symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms text cannot be empty.")

    from app.reasoning import generate_dpt_report_stream
    from app.config import DPT_TEXT_PROMPT_TEMPLATE

    prompt = DPT_TEXT_PROMPT_TEMPLATE.format(symptoms=symptoms.strip())

    async def event_stream():
        meta = {
            "type": "text_start",
            "symptoms": symptoms.strip()
        }
        yield f"data: {json.dumps(meta)}\n\n"

        async for token in generate_dpt_report_stream(prompt):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ──────────────────────────────────────
# Audio Analysis Endpoint
# ──────────────────────────────────────
@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (WAV, MP3, FLAC) with spoken symptoms.
    The audio is converted to text, then analyzed for a DPT report.
    """
    if not pipeline.is_loaded():
        raise HTTPException(status_code=503, detail="Models are still loading. Please wait.")

    allowed_types = {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3",
        "audio/flac", "audio/x-flac",
        "audio/ogg", "audio/webm",
    }
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio type: {file.content_type}. Accepted: WAV, MP3, FLAC, OGG",
        )

    start = time.time()
    audio_bytes = await file.read()

    result = await pipeline.process_audio(audio_bytes, file.filename or "audio.wav")
    result["processing_time_seconds"] = round(time.time() - start, 2)

    return JSONResponse(content=result)

@app.post("/analyze/audio/stream")
async def analyze_audio_stream(file: UploadFile = File(...)):
    if not pipeline.is_loaded():
        raise HTTPException(status_code=503, detail="Models are still loading.")
    
    audio_bytes = await file.read()
    from app.speech import transcribe_audio
    from app.reasoning import generate_dpt_report_stream
    from app.config import DPT_TEXT_PROMPT_TEMPLATE

    transcription = transcribe_audio(audio_bytes, file.filename or "audio.wav")
    
    async def event_stream():
        if not transcription["success"]:
            yield f"data: {json.dumps({'type': 'error', 'message': transcription['error']})}\n\n"
            return
            
        symptoms = transcription["text"]
        prompt = DPT_TEXT_PROMPT_TEMPLATE.format(symptoms=symptoms)
        
        meta = {
            "type": "audio_start",
            "transcription": symptoms
        }
        yield f"data: {json.dumps(meta)}\n\n"

        async for token in generate_dpt_report_stream(prompt):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ──────────────────────────────────────
# Streaming Report Endpoint (optional)
# ──────────────────────────────────────
@app.post("/analyze/image/stream")
async def analyze_image_stream(file: UploadFile = File(...)):
    """
    Same as /analyze/image but streams the DPT report in real-time.
    Useful for showing generation progress in the UI.
    """
    if not pipeline.is_loaded():
        raise HTTPException(status_code=503, detail="Models are still loading.")

    image_bytes = await file.read()
    from app.model_loader import load_image_from_bytes
    from app.config import DPT_IMAGE_PROMPT_TEMPLATE
    from app.reasoning import generate_dpt_report_stream

    image = load_image_from_bytes(image_bytes)

    # Route & classify
    routing = pipeline.classify_image_domain(image)
    classification = pipeline.classify_disease(image, routing["domain"])

    domain_full = {
        "eye": "Ophthalmology (Eye)",
        "dental": "Dentistry (Oral Health)",
        "skin": "Dermatology (Skin)",
    }

    prompt = DPT_IMAGE_PROMPT_TEMPLATE.format(
        domain=routing["domain"],
        disease=classification["disease"],
        confidence=classification["confidence"],
        domain_full=domain_full.get(routing["domain"], routing["domain"]),
    )

    # Stream the metadata first, then the report
    async def event_stream():
        # Send classification info first
        meta = {
            "type": "classification",
            "routing": routing,
            "classification": classification,
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # Stream report tokens
        async for token in generate_dpt_report_stream(prompt):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ──────────────────────────────────────
# Run with: python main.py
# ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
