"""
LLM reasoning engine (OpenAI-compatible backend).
Works with LM Studio, Ollama (/v1), vLLM, or any OpenAI-compatible server.
Generates DPT (Diagnostic, Prognostic, Therapeutic) clinical reports.
"""

import httpx
import json
from app.config import LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, DPT_SYSTEM_PROMPT


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }


def _build_payload(prompt: str, stream: bool) -> dict:
    return {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": DPT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stream": stream,
    }


async def check_llm_status() -> bool:
    """Check if the LLM backend is running and the configured model is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LLM_BASE_URL}/models", headers=_headers())
            if resp.status_code == 200:
                model_ids = [m.get("id") for m in resp.json().get("data", [])]
                return LLM_MODEL in model_ids
    except Exception:
        return False
    return False


# Backwards-compatible alias (older code called this check_ollama_status).
check_ollama_status = check_llm_status


async def generate_dpt_report(prompt: str) -> str:
    """
    Send a prompt to the LLM and return the complete generated text.
    """
    payload = _build_payload(prompt, stream=False)

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get(
                    "content", "Error: No response generated."
                )
            return "Error: No response generated."

    except httpx.TimeoutException:
        return "Error: LLM request timed out. Please ensure the model is loaded."
    except httpx.ConnectError:
        return (
            "Error: Cannot connect to the LLM backend. "
            f"Please ensure your server is running at {LLM_BASE_URL}."
        )
    except Exception as e:
        return f"Error generating report: {str(e)}"


async def generate_dpt_report_stream(prompt: str):
    """
    Stream the DPT report token-by-token for real-time UI updates.
    Yields chunks of text as they are generated.
    """
    payload = _build_payload(prompt, stream=True)

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST",
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                headers=_headers(),
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    # OpenAI-style SSE: lines are prefixed with "data: "
                    if line.startswith("data: "):
                        line = line[len("data: "):]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue

    except httpx.ConnectError:
        yield (
            "Error: Cannot connect to the LLM backend. "
            f"Please ensure your server is running at {LLM_BASE_URL}."
        )
    except Exception as e:
        yield f"Error: {str(e)}"
