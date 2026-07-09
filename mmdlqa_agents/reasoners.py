from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mmdlqa_core.config import Settings
from mmdlqa_core.contracts import AgentState, AnswerCandidate, ToolCallRecord
from mmdlqa_core.openrouter import OpenRouterClient
from mmdlqa_core.schema import AnswerResult, Question, RetrievedChunk
from mmdlqa_core.utils import dedupe_keep_order, json_dumps, normalize_text
from mmdlqa_retrieval.hybrid import top_evidence_files

from .answering import Answerer, normalize_answer


class ToolReasoner:
    def __init__(self, settings: Settings, answerer: Answerer):
        self.settings = settings
        self.answerer = answerer

    def run(self, state: AgentState) -> list[AnswerCandidate]:
        candidates: list[AnswerCandidate] = []
        vision = self.answerer.maybe_answer_image_group(state.question, state.evidence_pool)
        if vision:
            candidates.append(candidate_from_result("vision_tool", vision, confidence=0.9))
            state.tool_calls.append(
                ToolCallRecord(
                    tool_name="vision_group",
                    input={"question": state.question.question},
                    output={"answer": vision.answer, "evidences": vision.evidences},
                )
            )

        deterministic = self.answerer.try_deterministic(state.question, state.evidence_pool)
        if deterministic:
            candidates.append(candidate_from_result("deterministic_tool", deterministic, confidence=0.95))
            state.tool_calls.append(
                ToolCallRecord(
                    tool_name=str(deterministic.diagnostics.get("method", "deterministic")),
                    input={"question": state.question.question},
                    output={"answer": deterministic.answer, "evidences": deterministic.evidences},
                )
            )
        return candidates


class PromptedReasoner:
    def __init__(
        self,
        settings: Settings,
        answerer: Answerer,
        *,
        name: str,
        system_prompt: str,
        model: str | None = None,
        confidence: float = 0.7,
    ):
        self.settings = settings
        self.answerer = answerer
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.confidence = confidence
        self.llm = OpenRouterClient(settings)

    def run(self, state: AgentState) -> list[AnswerCandidate]:
        if not self.llm.available:
            return []
        try:
            candidate = self._run_llm(state)
            return [candidate] if candidate else []
        except Exception as exc:
            return [
                AnswerCandidate(
                    source=self.name,
                    answer="",
                    confidence=0.0,
                    diagnostics={"error": repr(exc), "failed": True},
                )
            ]

    def _run_llm(self, state: AgentState) -> AnswerCandidate | None:
        narrowed = self.answerer.rerank(state.question, state.evidence_pool)
        if not narrowed:
            return None
        exact = state.question.answer_type.casefold() == "exact_match"
        context = self.answerer.build_context(narrowed)
        payload = {
            "question": asdict(state.question),
            "reasoning_steps": [
                {
                    "step_id": step.step_id,
                    "purpose": step.purpose,
                    "sentence": step.sentence,
                    "depends_on": step.depends_on,
                }
                for step in state.steps
            ],
            "retrieval_queries": [
                {"step_id": query.step_id, "purpose": query.purpose, "sentence": query.sentence}
                for query in state.rag_queries
            ],
            "context": context,
            "exact_match_style": exact,
            "instructions": {
                "evidences_must_be_files_from_context": True,
                "insufficient_answer": "Not enough data to answer.",
            },
        }
        data = self.llm.json_chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json_dumps(payload)},
            ],
            model=self.model,
            max_tokens=1200,
        )
        answer = normalize_answer(str(data.get("answer", "")), exact)
        evidences = data.get("evidences", [])
        if not isinstance(evidences, list):
            evidences = []
        valid_files = {result.chunk.file_path for result in narrowed}
        evidences = [str(item) for item in evidences if str(item) in valid_files]
        if not evidences and answer and answer != "Not enough data to answer.":
            evidences = top_evidence_files(narrowed, self.settings.max_files_for_question)
        rationale = normalize_text(str(data.get("rationale", "")))
        return AnswerCandidate(
            source=self.name,
            answer=answer or "Not enough data to answer.",
            evidences=dedupe_keep_order(evidences),
            confidence=self.confidence,
            rationale=rationale,
            diagnostics={
                "method": "llm_reasoner",
                "model": self.model or self.settings.openrouter_model,
                "context_chunks": len(narrowed),
            },
        )


class FallbackReasoner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, state: AgentState) -> list[AnswerCandidate]:
        return [
            AnswerCandidate(
                source="fallback",
                answer="Not enough data to answer.",
                evidences=top_evidence_files(state.evidence_pool, self.settings.max_files_for_question),
                confidence=0.05,
                diagnostics={"method": "fallback"},
            )
        ]


class MultiExpertReasoner:
    def __init__(self, settings: Settings, answerer: Answerer):
        self.settings = settings
        self.answerer = answerer
        self.reasoners = build_reasoners(settings, answerer)

    def run(self, state: AgentState) -> list[AnswerCandidate]:
        candidates: list[AnswerCandidate] = []
        for reasoner in self.reasoners:
            candidates.extend(reasoner.run(state))
        return candidates


class CandidateAggregator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def choose(self, state: AgentState) -> AnswerCandidate:
        if not state.candidates:
            return AnswerCandidate(source="empty", answer="Not enough data to answer.", confidence=0.0)

        def score(candidate: AnswerCandidate) -> tuple[float, int, int, float]:
            has_answer = int(bool(candidate.answer and candidate.answer != "Not enough data to answer."))
            valid_evidence_count = sum(1 for item in candidate.evidences if self._valid_evidence_file(item, state))
            invalid_evidence_count = sum(1 for item in candidate.evidences if not self._valid_evidence_file(item, state))
            evidence_bonus = min(valid_evidence_count, self.settings.max_files_for_question) * 0.03
            invalid_penalty = invalid_evidence_count * 0.2
            answer_bonus = 0.2 if has_answer else 0.0
            return (
                candidate.confidence + evidence_bonus + answer_bonus - invalid_penalty,
                has_answer,
                valid_evidence_count,
                -invalid_evidence_count,
            )

        return max(state.candidates, key=score)

    def _valid_evidence_file(self, path: str, state: AgentState) -> bool:
        if path in {result.chunk.file_path for result in state.evidence_pool}:
            return True
        return (self.settings.raw_dir / path).exists()


def build_reasoners(settings: Settings, answerer: Answerer):
    reasoners = [ToolReasoner(settings, answerer)]
    if settings.use_agentic_moe:
        models = [item.strip() for item in settings.agentic_moe_models.split(",") if item.strip()]
        exact_model = models[0] if models else None
        synthesis_model = models[1] if len(models) > 1 else exact_model
        reasoners.extend(
            [
                PromptedReasoner(
                    settings,
                    answerer,
                    name="exact_reasoner",
                    model=exact_model,
                    confidence=0.72,
                    system_prompt=(
                        "You are an exact-answer specialist for data-lake QA. "
                        "Use only the provided context. Return JSON with keys answer, evidences, rationale. "
                        "For exact_match, answer with only the minimal value, label, option letter, date, or short phrase. "
                        "If the context is insufficient, answer exactly: Not enough data to answer."
                    ),
                ),
                PromptedReasoner(
                    settings,
                    answerer,
                    name="synthesis_reasoner",
                    model=synthesis_model,
                    confidence=0.68,
                    system_prompt=(
                        "You are a multi-hop synthesis specialist for data-lake QA. "
                        "Use the reasoning steps and context to combine evidence across files when needed. "
                        "Return JSON with keys answer, evidences, rationale. "
                        "Evidences must be file paths from the context. "
                        "If the context does not support the answer, answer exactly: Not enough data to answer."
                    ),
                ),
            ]
        )
    reasoners.append(FallbackReasoner(settings))
    return reasoners


def candidate_from_result(source: str, result: AnswerResult, *, confidence: float) -> AnswerCandidate:
    return AnswerCandidate(
        source=source,
        answer=result.answer,
        evidences=result.evidences,
        confidence=confidence,
        diagnostics=result.diagnostics,
    )
