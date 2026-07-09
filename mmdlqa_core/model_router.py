from __future__ import annotations

from .config import Settings
from .model_catalog import DEFAULT_ROLE_MODELS


ROLE_SETTING_FIELDS = {
    "planner": "planner_model",
    "rerank": "rerank_model",
    "exact": "exact_model",
    "synthesis": "synthesis_model",
    "critic": "critic_model",
    "coder": "coder_model",
    "vision": "vision_model",
    "scan_text": "scan_text_model",
    "scan_table": "scan_table_model",
    "scan_document": "scan_document_model",
    "scan_image": "scan_image_model",
    "scan_audio": "scan_audio_model",
    "scan_video": "scan_video_model",
}


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def model_for(self, role: str) -> str:
        if not self.settings.use_model_router:
            return self.settings.openrouter_model
        role_key = (role or "default").strip().casefold()
        field = ROLE_SETTING_FIELDS.get(role_key)
        if field:
            configured = str(getattr(self.settings, field, "") or "").strip()
            if configured:
                return configured
        fallback = DEFAULT_ROLE_MODELS.get(role_key, DEFAULT_ROLE_MODELS["default"])
        return str(fallback or self.settings.openrouter_model).strip()

    def snapshot(self) -> dict[str, str]:
        return {role: self.model_for(role) for role in ROLE_SETTING_FIELDS}
