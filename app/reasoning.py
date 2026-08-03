"""
Ollama / Gemma 3 reasoning engine.
Generates DPT (Diagnostic, Prognostic, Therapeutic) clinical reports.
"""

import httpx
import json
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, DPT_SYSTEM_PROMPT


async def check_ollama_status() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m["name"] for m in models]
                return OLLAMA_MODEL in model_names
    except Exception:
        return False
    return False


async def generate_dpt_report(prompt: str) -> str:
    """
    Send a prompt to Gemma 3 via Ollama and stream the response.
    Returns the complete generated text.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "system": DPT_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 2048,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "Error: No response generated.")

    except httpx.TimeoutException:
        return "Error: Ollama request timed out. Please ensure the model is loaded."
    except httpx.ConnectError:
        return "Error: Cannot connect to Ollama. Please ensure Ollama is running (ollama serve)."
    except Exception as e:
        return f"Error generating report: {str(e)}"


async def generate_dpt_report_stream(prompt: str):
    """
    Stream the DPT report token-by-token for real-time UI updates.
    Yields chunks of text as they are generated.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "system": DPT_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 2048,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

    except httpx.ConnectError:
        yield "Error: Cannot connect to Ollama. Please ensure Ollama is running."
    except Exception as e:
        yield f"Error: {str(e)}"
