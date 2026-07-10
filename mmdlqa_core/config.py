from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .model_catalog import DEFAULT_ROLE_MODELS


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    input_dir: Path = Path("input")
    raw_dir: Path = Path("input/raw")
    text_cleaning_output_dir: Path = Path("input/text_cleaning_output")
    questions_path: Path = Path("input/questions.xlsx")
    output_dir: Path = Path("output")
    cache_dir: Path = Path("output/cache")
    submission_path: Path = Path("output/submission.csv")
    max_questions: int = 5

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = DEFAULT_ROLE_MODELS["default"]
    openrouter_referer: str = "https://github.com/HuyThinhUET/MMDLQA"
    openrouter_app_name: str = "MMDLQA Agentic QA"

    use_llm: bool = True
    use_text_cleaning_output: bool = True
    include_raw_fallback: bool = True
    use_model_router: bool = True
    use_llm_summaries: bool = False
    use_llm_rerank: bool = True
    use_vision_llm: bool = True
    use_whisper: bool = True
    use_video_processing: bool = True
    use_sentence_transformers: bool = False
    use_question_classifier: bool = True
    use_agentic_planner: bool = True
    use_agentic_moe: bool = True
    use_agentic_critic: bool = True
    use_agentic_tools: bool = True
    use_agentic_coder: bool = True
    use_evidence_scanner: bool = True
    force_best_effort_answer: bool = True
    use_coder_planner: bool = False
    planner_model: str = DEFAULT_ROLE_MODELS["planner"]
    rerank_model: str = DEFAULT_ROLE_MODELS["rerank"]
    exact_model: str = DEFAULT_ROLE_MODELS["exact"]
    synthesis_model: str = DEFAULT_ROLE_MODELS["synthesis"]
    critic_model: str = DEFAULT_ROLE_MODELS["critic"]
    coder_model: str = DEFAULT_ROLE_MODELS["coder"]
    vision_model: str = DEFAULT_ROLE_MODELS["vision"]
    scan_text_model: str = DEFAULT_ROLE_MODELS["scan_text"]
    scan_table_model: str = DEFAULT_ROLE_MODELS["scan_table"]
    scan_document_model: str = DEFAULT_ROLE_MODELS["scan_document"]
    scan_image_model: str = DEFAULT_ROLE_MODELS["scan_image"]
    scan_audio_model: str = DEFAULT_ROLE_MODELS["scan_audio"]
    scan_video_model: str = DEFAULT_ROLE_MODELS["scan_video"]

    chunk_size_chars: int = 3200
    chunk_overlap_chars: int = 450
    retrieve_top_k: int = 10
    rerank_top_k: int = 8
    rerank_candidate_k: int = 24
    max_context_chars: int = 14000
    max_files_for_question: int = 8
    request_timeout_sec: int = 120
    max_image_side: int = 1280
    video_frame_count: int = 6
    agentic_max_steps: int = 4
    agentic_min_rounds: int = 2
    agentic_max_rounds: int = 10
    agentic_moe_models: str = ""
    evidence_scan_max_files: int = 12
    evidence_scan_chunks_per_file: int = 2
    evidence_scan_max_chars_per_file: int = 1600
    evidence_scan_batch_size: int = 12
    evidence_scan_irrelevant_patience: int = 5
    print_question_metrics: bool = False
    max_question_seconds: float = 0.0
    max_question_llm_calls: int = 0
    max_question_cost_usd: float = 0.08
    max_question_rag_queries: int = 0
    llm_input_cost_per_million_tokens: float = 0.0
    llm_output_cost_per_million_tokens: float = 0.0

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
            text_cleaning_output_dir=Path(
                os.getenv("MMDLQA_TEXT_CLEANING_OUTPUT_DIR", "input/text_cleaning_output")
            ),
            questions_path=Path(os.getenv("MMDLQA_QUESTIONS", "input/questions.xlsx")),
            output_dir=Path(os.getenv("MMDLQA_OUTPUT_DIR", "output")),
            cache_dir=Path(os.getenv("MMDLQA_CACHE_DIR", "output/cache")),
            submission_path=Path(os.getenv("MMDLQA_SUBMISSION", "output/submission.csv")),
            max_questions=int(os.getenv("MMDLQA_MAX_QUESTIONS", "5")),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", DEFAULT_ROLE_MODELS["default"]),
            openrouter_referer=os.getenv(
                "OPENROUTER_REFERER", "https://github.com/HuyThinhUET/MMDLQA"
            ),
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", "MMDLQA Agentic QA"),
            use_llm=_bool_env("MMDLQA_USE_LLM", True),
            use_text_cleaning_output=_bool_env("MMDLQA_USE_TEXT_CLEANING_OUTPUT", True),
            include_raw_fallback=_bool_env("MMDLQA_INCLUDE_RAW_FALLBACK", True),
            use_model_router=_bool_env("MMDLQA_USE_MODEL_ROUTER", True),
            use_llm_summaries=_bool_env("MMDLQA_USE_LLM_SUMMARIES", False),
            use_llm_rerank=_bool_env("MMDLQA_USE_LLM_RERANK", True),
            use_vision_llm=_bool_env("MMDLQA_USE_VISION_LLM", True),
            use_whisper=_bool_env("MMDLQA_USE_WHISPER", True),
            use_video_processing=_bool_env("MMDLQA_USE_VIDEO_PROCESSING", True),
            use_sentence_transformers=_bool_env("MMDLQA_USE_SENTENCE_TRANSFORMERS", False),
            use_question_classifier=_bool_env("MMDLQA_USE_QUESTION_CLASSIFIER", True),
            use_agentic_planner=_bool_env("MMDLQA_USE_AGENTIC_PLANNER", True),
            use_agentic_moe=_bool_env("MMDLQA_USE_AGENTIC_MOE", True),
            use_agentic_critic=_bool_env("MMDLQA_USE_AGENTIC_CRITIC", True),
            use_agentic_tools=_bool_env("MMDLQA_USE_AGENTIC_TOOLS", True),
            use_agentic_coder=_bool_env("MMDLQA_USE_AGENTIC_CODER", True),
            use_evidence_scanner=_bool_env("MMDLQA_USE_EVIDENCE_SCANNER", True),
            force_best_effort_answer=_bool_env("MMDLQA_FORCE_BEST_EFFORT_ANSWER", True),
            use_coder_planner=_bool_env("MMDLQA_USE_CODER_PLANNER", False),
            planner_model=os.getenv("MMDLQA_PLANNER_MODEL", DEFAULT_ROLE_MODELS["planner"]),
            rerank_model=os.getenv("MMDLQA_RERANK_MODEL", DEFAULT_ROLE_MODELS["rerank"]),
            exact_model=os.getenv("MMDLQA_EXACT_MODEL", DEFAULT_ROLE_MODELS["exact"]),
            synthesis_model=os.getenv("MMDLQA_SYNTHESIS_MODEL", DEFAULT_ROLE_MODELS["synthesis"]),
            critic_model=os.getenv("MMDLQA_CRITIC_MODEL", DEFAULT_ROLE_MODELS["critic"]),
            coder_model=os.getenv("MMDLQA_CODER_MODEL", DEFAULT_ROLE_MODELS["coder"]),
            vision_model=os.getenv("MMDLQA_VISION_MODEL", DEFAULT_ROLE_MODELS["vision"]),
            scan_text_model=os.getenv("MMDLQA_SCAN_TEXT_MODEL", DEFAULT_ROLE_MODELS["scan_text"]),
            scan_table_model=os.getenv("MMDLQA_SCAN_TABLE_MODEL", DEFAULT_ROLE_MODELS["scan_table"]),
            scan_document_model=os.getenv("MMDLQA_SCAN_DOCUMENT_MODEL", DEFAULT_ROLE_MODELS["scan_document"]),
            scan_image_model=os.getenv("MMDLQA_SCAN_IMAGE_MODEL", DEFAULT_ROLE_MODELS["scan_image"]),
            scan_audio_model=os.getenv("MMDLQA_SCAN_AUDIO_MODEL", DEFAULT_ROLE_MODELS["scan_audio"]),
            scan_video_model=os.getenv("MMDLQA_SCAN_VIDEO_MODEL", DEFAULT_ROLE_MODELS["scan_video"]),
            chunk_size_chars=int(os.getenv("MMDLQA_CHUNK_SIZE_CHARS", "3200")),
            chunk_overlap_chars=int(os.getenv("MMDLQA_CHUNK_OVERLAP_CHARS", "450")),
            retrieve_top_k=int(os.getenv("MMDLQA_RETRIEVE_TOP_K", "10")),
            rerank_top_k=int(os.getenv("MMDLQA_RERANK_TOP_K", "8")),
            rerank_candidate_k=int(os.getenv("MMDLQA_RERANK_CANDIDATE_K", "24")),
            max_context_chars=int(os.getenv("MMDLQA_MAX_CONTEXT_CHARS", "14000")),
            max_files_for_question=int(os.getenv("MMDLQA_MAX_FILES_FOR_QUESTION", "8")),
            request_timeout_sec=int(os.getenv("MMDLQA_REQUEST_TIMEOUT_SEC", "120")),
            max_image_side=int(os.getenv("MMDLQA_MAX_IMAGE_SIDE", "1280")),
            video_frame_count=int(os.getenv("MMDLQA_VIDEO_FRAME_COUNT", "6")),
            agentic_max_steps=int(os.getenv("MMDLQA_AGENTIC_MAX_STEPS", "4")),
            agentic_min_rounds=int(os.getenv("MMDLQA_AGENTIC_MIN_ROUNDS", "2")),
            agentic_max_rounds=int(os.getenv("MMDLQA_AGENTIC_MAX_ROUNDS", "10")),
            agentic_moe_models=os.getenv("MMDLQA_AGENTIC_MOE_MODELS", ""),
            evidence_scan_max_files=int(os.getenv("MMDLQA_EVIDENCE_SCAN_MAX_FILES", "12")),
            evidence_scan_chunks_per_file=int(os.getenv("MMDLQA_EVIDENCE_SCAN_CHUNKS_PER_FILE", "2")),
            evidence_scan_max_chars_per_file=int(os.getenv("MMDLQA_EVIDENCE_SCAN_MAX_CHARS_PER_FILE", "1600")),
            evidence_scan_batch_size=int(os.getenv("MMDLQA_EVIDENCE_SCAN_BATCH_SIZE", "12")),
            evidence_scan_irrelevant_patience=int(os.getenv("MMDLQA_EVIDENCE_SCAN_IRRELEVANT_PATIENCE", "5")),
            print_question_metrics=_bool_env("MMDLQA_PRINT_QUESTION_METRICS", False),
            max_question_seconds=_float_env("MMDLQA_MAX_QUESTION_SECONDS", 0.0),
            max_question_llm_calls=int(os.getenv("MMDLQA_MAX_QUESTION_LLM_CALLS", "0")),
            max_question_cost_usd=_float_env("MMDLQA_MAX_QUESTION_COST_USD", 0.08),
            max_question_rag_queries=int(os.getenv("MMDLQA_MAX_QUESTION_RAG_QUERIES", "0")),
            llm_input_cost_per_million_tokens=_float_env("MMDLQA_LLM_INPUT_COST_PER_MILLION_TOKENS", 0.0),
            llm_output_cost_per_million_tokens=_float_env("MMDLQA_LLM_OUTPUT_COST_PER_MILLION_TOKENS", 0.0),
        )

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
