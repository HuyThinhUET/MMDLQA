from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True)
class Settings:
    input_dir: Path = Path("input")
    raw_dir: Path = Path("input/raw")
    questions_path: Path = Path("input/sample_questions.xlsx")
    output_dir: Path = Path("output")
    cache_dir: Path = Path("output/cache")
    submission_path: Path = Path("output/submission.csv")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "gemma-4-26b-a4b-it"
    openrouter_referer: str = "https://github.com/HuyThinhUET/MMDLQA"
    openrouter_app_name: str = "MMDLQA Baseline"

    use_llm: bool = True
    use_llm_summaries: bool = False
    use_llm_rerank: bool = True
    use_vision_llm: bool = True
    use_whisper: bool = True
    use_sentence_transformers: bool = False

    chunk_size_chars: int = 3200
    chunk_overlap_chars: int = 450
    retrieve_top_k: int = 12
    rerank_top_k: int = 8
    max_context_chars: int = 24000
    max_files_for_question: int = 8
    request_timeout_sec: int = 120
    max_image_side: int = 1280
    video_frame_count: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        return cls(
            input_dir=Path(os.getenv("MMDLQA_INPUT_DIR", "input")),
            raw_dir=Path(os.getenv("MMDLQA_RAW_DIR", "input/raw")),
            questions_path=Path(os.getenv("MMDLQA_QUESTIONS", "input/sample_questions.xlsx")),
            output_dir=Path(os.getenv("MMDLQA_OUTPUT_DIR", "output")),
            cache_dir=Path(os.getenv("MMDLQA_CACHE_DIR", "output/cache")),
            submission_path=Path(os.getenv("MMDLQA_SUBMISSION", "output/submission.csv")),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "gemma-4-26b-a4b-it"),
            openrouter_referer=os.getenv(
                "OPENROUTER_REFERER", "https://github.com/HuyThinhUET/MMDLQA"
            ),
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", "MMDLQA Baseline"),
            use_llm=_bool_env("MMDLQA_USE_LLM", True),
            use_llm_summaries=_bool_env("MMDLQA_USE_LLM_SUMMARIES", False),
            use_llm_rerank=_bool_env("MMDLQA_USE_LLM_RERANK", True),
            use_vision_llm=_bool_env("MMDLQA_USE_VISION_LLM", True),
            use_whisper=_bool_env("MMDLQA_USE_WHISPER", True),
            use_sentence_transformers=_bool_env("MMDLQA_USE_SENTENCE_TRANSFORMERS", False),
            chunk_size_chars=int(os.getenv("MMDLQA_CHUNK_SIZE_CHARS", "3200")),
            chunk_overlap_chars=int(os.getenv("MMDLQA_CHUNK_OVERLAP_CHARS", "450")),
            retrieve_top_k=int(os.getenv("MMDLQA_RETRIEVE_TOP_K", "12")),
            rerank_top_k=int(os.getenv("MMDLQA_RERANK_TOP_K", "8")),
            max_context_chars=int(os.getenv("MMDLQA_MAX_CONTEXT_CHARS", "24000")),
            max_files_for_question=int(os.getenv("MMDLQA_MAX_FILES_FOR_QUESTION", "8")),
            request_timeout_sec=int(os.getenv("MMDLQA_REQUEST_TIMEOUT_SEC", "120")),
            max_image_side=int(os.getenv("MMDLQA_MAX_IMAGE_SIDE", "1280")),
            video_frame_count=int(os.getenv("MMDLQA_VIDEO_FRAME_COUNT", "6")),
        )

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
