from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from .config import Settings
from .model_catalog import pricing_for_model


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class StageMetric:
    name: str
    elapsed_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LlmCallMetric:
    model: str
    elapsed_sec: float
    ok: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    native_cost_usd: float | None = None
    estimated_cost_usd: float = 0.0
    error: str = ""


class QuestionRunTracker:
    def __init__(self, settings: Settings, qid: str):
        self.settings = settings
        self.qid = qid
        self.started_at = time.time()
        self._start_perf = time.perf_counter()
        self.elapsed_sec = 0.0
        self.stages: list[StageMetric] = []
        self.llm_calls: list[LlmCallMetric] = []
        self.limit_exceeded = False
        self.limit_reason = ""
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "QuestionRunTracker":
        self._token = _CURRENT_TRACKER.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.refresh_elapsed()
        if self._token is not None:
            _CURRENT_TRACKER.reset(self._token)

    def refresh_elapsed(self) -> float:
        self.elapsed_sec = round(time.perf_counter() - self._start_perf, 4)
        return self.elapsed_sec

    @contextlib.contextmanager
    def stage(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages.append(
                StageMetric(
                    name=name,
                    elapsed_sec=round(time.perf_counter() - start, 4),
                    metadata=metadata or {},
                )
            )

    def check_limits(self, where: str = "") -> None:
        if self.settings.max_question_seconds > 0 and self.refresh_elapsed() >= self.settings.max_question_seconds:
            self.mark_limit(f"time_limit_exceeded at {where}".strip())
        if (
            self.settings.max_question_llm_calls > 0
            and len(self.llm_calls) >= self.settings.max_question_llm_calls
        ):
            self.mark_limit(f"llm_call_limit_exceeded at {where}".strip())
        if (
            self.settings.max_question_cost_usd > 0
            and self.total_estimated_cost_usd >= self.settings.max_question_cost_usd
        ):
            self.mark_limit(f"cost_limit_exceeded at {where}".strip())

    def mark_limit(self, reason: str) -> None:
        self.note_limit(reason)
        raise BudgetExceededError(reason)

    def note_limit(self, reason: str) -> None:
        self.limit_exceeded = True
        self.limit_reason = reason

    def record_llm_call(
        self,
        *,
        model: str,
        elapsed_sec: float,
        ok: bool,
        usage: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        usage = usage or {}
        prompt_tokens = int_value(
            usage.get("prompt_tokens", usage.get("input_tokens", usage.get("prompt", 0)))
        )
        completion_tokens = int_value(
            usage.get("completion_tokens", usage.get("output_tokens", usage.get("completion", 0)))
        )
        total_tokens = int_value(usage.get("total_tokens", prompt_tokens + completion_tokens))
        native_cost = float_or_none(
            usage.get("cost", usage.get("total_cost", usage.get("cost_usd", None)))
        )
        estimated_cost = estimate_cost_usd(
            self.settings,
            prompt_tokens,
            completion_tokens,
            native_cost,
            model,
        )
        self.llm_calls.append(
            LlmCallMetric(
                model=model,
                elapsed_sec=round(elapsed_sec, 4),
                ok=ok,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                native_cost_usd=native_cost,
                estimated_cost_usd=round(estimated_cost, 8),
                error=error,
            )
        )

    @property
    def total_prompt_tokens(self) -> int:
        return sum(call.prompt_tokens for call in self.llm_calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(call.completion_tokens for call in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for call in self.llm_calls)

    @property
    def total_estimated_cost_usd(self) -> float:
        return round(sum(call.estimated_cost_usd for call in self.llm_calls), 8)

    def snapshot(self) -> dict[str, Any]:
        self.refresh_elapsed()
        return {
            "qid": self.qid,
            "started_at": self.started_at,
            "elapsed_sec": self.elapsed_sec,
            "stage_count": len(self.stages),
            "stages": [asdict(stage) for stage in self.stages],
            "llm_call_count": len(self.llm_calls),
            "llm_calls": [asdict(call) for call in self.llm_calls],
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "limit_exceeded": self.limit_exceeded,
            "limit_reason": self.limit_reason,
            "limits": {
                "max_question_seconds": self.settings.max_question_seconds,
                "max_question_llm_calls": self.settings.max_question_llm_calls,
                "max_question_cost_usd": self.settings.max_question_cost_usd,
                "max_question_rag_queries": self.settings.max_question_rag_queries,
            },
        }


_CURRENT_TRACKER: contextvars.ContextVar[QuestionRunTracker | None] = contextvars.ContextVar(
    "mmdlqa_question_run_tracker",
    default=None,
)


def current_tracker() -> QuestionRunTracker | None:
    return _CURRENT_TRACKER.get()


def estimate_cost_usd(
    settings: Settings,
    prompt_tokens: int,
    completion_tokens: int,
    native_cost: float | None,
    model: str = "",
) -> float:
    if native_cost is not None:
        return native_cost
    model_pricing = pricing_for_model(model)
    if model_pricing:
        return prompt_tokens * model_pricing.prompt + completion_tokens * model_pricing.completion
    input_cost = prompt_tokens * settings.llm_input_cost_per_million_tokens / 1_000_000
    output_cost = completion_tokens * settings.llm_output_cost_per_million_tokens / 1_000_000
    return input_cost + output_cost


def int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_question_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row.get("answer", {}).get("diagnostics", {}).get("metrics", {}) for row in rows]
    metrics = [m for m in metrics if isinstance(m, dict)]
    return {
        "question_count": len(metrics),
        "total_elapsed_sec": round(sum(float(m.get("elapsed_sec", 0.0)) for m in metrics), 4),
        "total_llm_calls": sum(int(m.get("llm_call_count", 0)) for m in metrics),
        "total_prompt_tokens": sum(int(m.get("total_prompt_tokens", 0)) for m in metrics),
        "total_completion_tokens": sum(int(m.get("total_completion_tokens", 0)) for m in metrics),
        "total_tokens": sum(int(m.get("total_tokens", 0)) for m in metrics),
        "total_estimated_cost_usd": round(
            sum(float(m.get("total_estimated_cost_usd", 0.0)) for m in metrics),
            8,
        ),
        "limit_exceeded_count": sum(1 for m in metrics if m.get("limit_exceeded")),
    }
