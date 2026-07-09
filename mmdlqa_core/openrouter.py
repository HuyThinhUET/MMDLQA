from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any

import requests

from .config import Settings
from .metrics import current_tracker


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
        tracker = current_tracker()
        if tracker:
            tracker.check_limits("before_llm_call")
        url = f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions"
        selected_model = model or self.settings.openrouter_model
        payload: dict[str, Any] = {
            "model": selected_model,
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
        start = time.perf_counter()
        usage: dict[str, Any] = {}
        actual_model = selected_model
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.settings.request_timeout_sec,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:1000]}")
            data = resp.json()
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            actual_model = str(data.get("model") or selected_model)
            content = data["choices"][0]["message"].get("content", "")
            usage = ensure_usage_estimate(usage, messages, content)
        except Exception as exc:
            if tracker:
                tracker.record_llm_call(
                    model=actual_model,
                    elapsed_sec=time.perf_counter() - start,
                    ok=False,
                    usage=usage,
                    error=repr(exc),
                )
            raise
        if tracker:
            tracker.record_llm_call(
                model=actual_model,
                elapsed_sec=time.perf_counter() - start,
                ok=True,
                usage=usage,
            )
            tracker.check_limits("after_llm_call")
        return content

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


def ensure_usage_estimate(
    usage: dict[str, Any],
    messages: list[dict[str, Any]],
    content: str,
) -> dict[str, Any]:
    prompt_tokens = int_token_value(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    completion_tokens = int_token_value(usage.get("completion_tokens", usage.get("output_tokens", 0)))
    total_tokens = int_token_value(usage.get("total_tokens", prompt_tokens + completion_tokens))
    if total_tokens > 0:
        return usage
    prompt_tokens = estimate_tokens(json.dumps(messages, ensure_ascii=False))
    completion_tokens = estimate_tokens(content)
    return {
        **usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "usage_estimated": True,
    }


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text or "") / 4))


def int_token_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
