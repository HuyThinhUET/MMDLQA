from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .schema import AnswerResult, Question, RetrievedChunk


@dataclass(slots=True)
class RagQuery:
    sentence: str
    purpose: str = "source_retrieval"
    step_id: str = ""
    top_k: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReasoningStep:
    step_id: str
    sentence: str
    purpose: str = "answer_question"
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerCandidate:
    source: str
    answer: str
    evidences: list[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CriticReport:
    reviewer: str
    ok: bool
    issues: list[str] = field(default_factory=list)
    missing_queries: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: str
    input: dict[str, Any]
    output: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""


@dataclass(slots=True)
class AgentState:
    question: Question
    steps: list[ReasoningStep] = field(default_factory=list)
    rag_queries: list[RagQuery] = field(default_factory=list)
    evidence_pool: list[RetrievedChunk] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    candidates: list[AnswerCandidate] = field(default_factory=list)
    critic_reports: list[CriticReport] = field(default_factory=list)
    final_answer: AnswerResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SentenceRetriever(Protocol):
    def search_sentence(
        self,
        query: str | RagQuery,
        *,
        raw_dir: Path | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        ...
