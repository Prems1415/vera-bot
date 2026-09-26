"""Thin async LLM client. Supports Gemini and OpenAI-compatible APIs via env vars.

Env:
  LLM_PROVIDER   gemini | openai | groq | openrouter | deepseek   (default: gemini)
  LLM_API_KEY    the key (never hard-code / commit)
  LLM_MODEL      optional override
  USE_LLM        "false" to force deterministic templates only
"""
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
    if not enabled():
        return None
    try:
        if PROVIDER == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1200,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
            r = await _http().post(url, json=body, headers={"x-goog-api-key": API_KEY}, timeout=timeout)
            if r.status_code == 400 and "thinking" in r.text.lower():
                body["generationConfig"].pop("thinkingConfig", None)
                r = await _http().post(url, json=body, headers={"x-goog-api-key": API_KEY}, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
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
    except Exception as e:  # network, timeout, quota, parse — caller falls back
        print(f"[llm] failure: {type(e).__name__}: {str(e)[:200]}")
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
