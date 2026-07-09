from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import requests

from .config import Settings


class OpenRouterClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def available(self) -> bool:
        return bool(self.settings.openrouter_api_key and self.settings.use_llm)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> str:
        if not self.available:
            raise RuntimeError("OPENROUTER_API_KEY is not set or MMDLQA_USE_LLM=0.")
        url = f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": model or self.settings.openrouter_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_referer,
            "X-Title": self.settings.openrouter_app_name,
        }
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.settings.request_timeout_sec,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:1000]}")
        data = resp.json()
        return data["choices"][0]["message"].get("content", "")

    def json_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        text = self.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise


def image_part_from_path(path: Path, max_side: int = 1280) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = path.read_bytes()
    if mime in {"image/png", "image/jpeg", "image/webp"}:
        pil = None
        try:
            from PIL import Image
            import io

            pil = Image.open(io.BytesIO(data))
            pil.thumbnail((max_side, max_side))
            out = io.BytesIO()
            fmt = "JPEG" if mime == "image/jpeg" else "PNG"
            pil.save(out, format=fmt)
            data = out.getvalue()
        except Exception:
            pass
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }
