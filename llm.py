"""Thin async LLM client. Supports Gemini and OpenAI-compatible APIs via env vars."""
from __future__ import annotations

import json
import os
import re

import httpx

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
API_KEY = os.getenv("LLM_API_KEY", "").strip()
USE_LLM = os.getenv("USE_LLM", "true").strip().lower() not in ("0", "false", "no")

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "openai/gpt-4o-mini",
    "deepseek": "deepseek-chat",
}
MODEL = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS.get(PROVIDER, "gpt-4o-mini")

OPENAI_COMPAT = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
}

_client: httpx.AsyncClient | None = None
LAST_ERROR = ""
_GOOD_URL: str | None = None


def enabled() -> bool:
    return USE_LLM and bool(API_KEY)


def model_name() -> str:
    return MODEL if enabled() else "deterministic-templates"


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=4.0))
    return _client


async def complete_json(system: str, user: str, timeout: float = 9.0) -> dict | None:
    """Return parsed JSON object from the model, or None on any failure."""
    global LAST_ERROR
    if not enabled():
        return None
    try:
        if PROVIDER == "gemini":
            text = await _gemini(system, user, timeout)
            if text is None:
                return None
        else:
            url = OPENAI_COMPAT.get(PROVIDER, OPENAI_COMPAT["openai"])
            body = {
                "model": MODEL,
                "temperature": 0.2,
                "max_tokens": 900,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            r = await _http().post(url, json=body, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        return _parse(text)
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"[llm] failure: {LAST_ERROR}")
        return None


def _gemini_urls() -> list[str]:
    studio = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    vertex = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{MODEL}:generateContent"
    urls = [studio, vertex]
    if _GOOD_URL:
        urls = [_GOOD_URL] + [u for u in urls if u != _GOOD_URL]
    return urls


async def _gemini(system: str, user: str, timeout: float) -> str | None:
    global LAST_ERROR, _GOOD_URL
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1800,
                             "responseMimeType": "application/json",
                             "thinkingConfig": {"thinkingBudget": 0}},
    }
    for url in _gemini_urls():
        for with_thinking in (True, False):
            b = json.loads(json.dumps(body))
            if not with_thinking:
                b["generationConfig"].pop("thinkingConfig", None)
            r = await _http().post(url, json=b, headers={"x-goog-api-key": API_KEY}, timeout=timeout)
            if r.status_code == 200:
                _GOOD_URL = url
                data = r.json()
                parts = data["candidates"][0]["content"].get("parts", [])
                return "".join(p.get("text", "") for p in parts)
            LAST_ERROR = f"{url.split('/')[2]} HTTP {r.status_code}: {r.text[:250]}"
            print(f"[llm] {LAST_ERROR}")
            if r.status_code == 400 and "thinking" in r.text.lower() and with_thinking:
                continue
            break
    return None


def _parse(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return None
    return None
