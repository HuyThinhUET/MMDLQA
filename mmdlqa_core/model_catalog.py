from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ROLE_MODELS = {
    "default": "google/gemini-2.5-flash-lite",
    "planner": "google/gemini-2.5-flash-lite",
    "rerank": "google/gemini-2.5-flash-lite",
    "exact": "deepseek/deepseek-chat-v3.1",
    "synthesis": "deepseek/deepseek-chat-v3.1",
    "critic": "deepseek/deepseek-chat-v3.1",
    "coder": "qwen/qwen3-coder-flash",
    "vision": "google/gemini-2.5-flash",
    "scan_text": "google/gemini-2.5-flash-lite",
    "scan_table": "qwen/qwen3-coder-flash",
    "scan_document": "deepseek/deepseek-chat-v3.1",
    "scan_image": "google/gemini-2.5-flash",
    "scan_audio": "google/gemini-2.5-flash-lite",
    "scan_video": "google/gemini-2.5-flash-lite",
}


@dataclass(frozen=True, slots=True)
class TokenPricing:
    prompt: float
    completion: float


MODEL_PRICING_USD_PER_TOKEN = {
    "google/gemini-2.5-flash-lite": TokenPricing(prompt=0.10 / 1_000_000, completion=0.40 / 1_000_000),
    "google/gemini-2.5-flash": TokenPricing(prompt=0.30 / 1_000_000, completion=2.50 / 1_000_000),
    "deepseek/deepseek-chat-v3.1": TokenPricing(prompt=0.21 / 1_000_000, completion=0.79 / 1_000_000),
    "qwen/qwen3-coder-flash": TokenPricing(prompt=0.195 / 1_000_000, completion=0.975 / 1_000_000),
    "openai/gpt-5.6-luna": TokenPricing(prompt=1.00 / 1_000_000, completion=6.00 / 1_000_000),
    "openai/gpt-5.6-luna-pro": TokenPricing(prompt=1.00 / 1_000_000, completion=6.00 / 1_000_000),
    "openai/gpt-5.6-terra": TokenPricing(prompt=2.50 / 1_000_000, completion=15.00 / 1_000_000),
    "openai/gpt-5.6-terra-pro": TokenPricing(prompt=2.50 / 1_000_000, completion=15.00 / 1_000_000),
    "openai/gpt-5.6-sol": TokenPricing(prompt=5.00 / 1_000_000, completion=30.00 / 1_000_000),
    "openai/gpt-5.6-sol-pro": TokenPricing(prompt=5.00 / 1_000_000, completion=30.00 / 1_000_000),
    "anthropic/claude-sonnet-5": TokenPricing(prompt=2.00 / 1_000_000, completion=10.00 / 1_000_000),
    "x-ai/grok-4.5": TokenPricing(prompt=2.00 / 1_000_000, completion=6.00 / 1_000_000),
}


def pricing_for_model(model: str) -> TokenPricing | None:
    key = normalize_model_slug(model)
    if key.endswith(":free"):
        return TokenPricing(prompt=0.0, completion=0.0)
    return MODEL_PRICING_USD_PER_TOKEN.get(key) or MODEL_PRICING_USD_PER_TOKEN.get(
        key.split(":", 1)[0]
    )


def normalize_model_slug(model: str) -> str:
    return (model or "").strip().casefold()
