from __future__ import annotations

import re
from dataclasses import asdict

from mmdlqa_core.config import Settings
from mmdlqa_core.contracts import AgentState, AnswerCandidate
from mmdlqa_core.metrics import BudgetExceededError
from mmdlqa_core.model_router import ModelRouter
from mmdlqa_core.openrouter import OpenRouterClient
from mmdlqa_core.prompting import (
    answer_contract_payload,
    detect_answer_contract,
    secure_system_prompt,
    untrusted_data_notice,
)
from mmdlqa_core.utils import dedupe_keep_order, json_dumps, normalize_text, tokenize
from mmdlqa_retrieval.hybrid import top_evidence_files

from .answering import Answerer, normalize_answer
from .evidence import ledger_from_llm_output, supported_claim_count
from .structured import json_chat_validated, validate_answer_output
from .tool_agents import CoderAgent, ToolAgent


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
        except BudgetExceededError:
            raise
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
            "answer_contract": answer_contract_payload(state.question),
            "prompt_security": untrusted_data_notice(),
            "instructions": {
                "evidences_must_be_files_from_context": True,
                "claims_required": True,
                "best_effort_required": self.settings.force_best_effort_answer,
                "claims_schema": {
                    "claim": "one atomic statement supporting the answer",
                    "evidence_files": ["file paths from context supporting this claim"],
                    "quotes": ["short quote or data snippet when available"],
                },
                "insufficient_answer": (
                    "Use Not enough data to answer only if none of the provided context files contain "
                    "any plausible direct or partial evidence. If at least one file is plausibly relevant, "
                    "give the best supported answer and cite that file, even with low confidence."
                ),
            },
        }
        valid_files = {result.chunk.file_path for result in narrowed}
        schema_hint = {
            "answer": "string",
            "evidences": ["file paths from context"],
            "rationale": "short rationale",
            "claims": [
                {
                    "claim": "atomic claim",
                    "evidence_files": ["file paths from context"],
                    "quotes": ["short quotes or snippets"],
                }
            ],
        }
        validated = json_chat_validated(
            self.llm,
            [
                {
                    "role": "system",
                    "content": secure_system_prompt(
                        self.system_prompt,
                        state.question,
                        include_answer_contract=True,
                    ),
                },
                {"role": "user", "content": json_dumps(payload)},
            ],
            validator=validate_answer_output(
                valid_files,
                require_claims=True,
                allow_insufficient=not self.settings.force_best_effort_answer or not valid_files,
            ),
            schema_name=f"{self.name}_answer",
            schema_hint=schema_hint,
            model=self.model,
            max_tokens=1200,
            repair_max_tokens=800,
        )
        data = validated.data
        answer = normalize_answer(str(data.get("answer", "")), exact, state.question)
        evidences = data.get("evidences", [])
        if not isinstance(evidences, list):
            evidences = []
        evidences = [str(item) for item in evidences if str(item) in valid_files]
        if not evidences and answer and answer != "Not enough data to answer.":
            evidences = top_evidence_files(narrowed, self.settings.max_files_for_question)
        rationale = normalize_text(str(data.get("rationale", "")))
        claim_evidence = ledger_from_llm_output(self.name, data, answer, evidences, narrowed)
        return AnswerCandidate(
            source=self.name,
            answer=answer or "Not enough data to answer.",
            evidences=dedupe_keep_order(evidences),
            claim_evidence=claim_evidence,
            confidence=self.confidence,
            rationale=rationale,
            diagnostics={
                "method": "llm_reasoner",
                "model": self.model or self.settings.openrouter_model,
                "context_chunks": len(narrowed),
                "validation": validated.diagnostics,
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
                claim_evidence=[],
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
            return self.ensure_best_effort(
                state,
                AnswerCandidate(source="empty", answer="Not enough data to answer.", confidence=0.0),
            )

        def score(candidate: AnswerCandidate) -> tuple[float, int, int, int, float]:
            has_answer = int(bool(candidate.answer and candidate.answer != "Not enough data to answer."))
            valid_evidence_count = sum(1 for item in candidate.evidences if self._valid_evidence_file(item, state))
            invalid_evidence_count = sum(1 for item in candidate.evidences if not self._valid_evidence_file(item, state))
            supported_claims = supported_claim_count(candidate.claim_evidence)
            evidence_bonus = min(valid_evidence_count, self.settings.max_files_for_question) * 0.03
            claim_bonus = min(supported_claims, 4) * 0.06
            invalid_penalty = invalid_evidence_count * 0.2
            answer_bonus = 0.2 if has_answer else 0.0
            return (
                candidate.confidence + evidence_bonus + claim_bonus + answer_bonus - invalid_penalty,
                has_answer,
                supported_claims,
                valid_evidence_count,
                -invalid_evidence_count,
            )

        return self.ensure_best_effort(state, max(state.candidates, key=score))

    def ensure_best_effort(self, state: AgentState, candidate: AnswerCandidate) -> AnswerCandidate:
        if not self.settings.force_best_effort_answer:
            return candidate
        if candidate.answer and candidate.answer != "Not enough data to answer.":
            return candidate
        evidence_files = top_evidence_files(state.evidence_pool, self.settings.max_files_for_question)
        evidence_files = [path for path in evidence_files if self._valid_evidence_file(path, state)]
        if not evidence_files:
            return candidate
        alternative = best_non_empty_candidate(state, self)
        if alternative:
            return alternative
        return AnswerCandidate(
            source="best_effort_static",
            answer=best_effort_static_answer(state),
            evidences=evidence_files,
            claim_evidence=[],
            confidence=0.18,
            rationale="Forced best-effort answer because valid retrieved evidence exists.",
            diagnostics={
                "method": "best_effort_static",
                "forced_after_insufficient": True,
                "evidence_files": evidence_files,
            },
        )

    def _valid_evidence_file(self, path: str, state: AgentState) -> bool:
        if path in {result.chunk.file_path for result in state.evidence_pool}:
            return True
        return (self.settings.raw_dir / path).exists()


def build_reasoners(settings: Settings, answerer: Answerer):
    reasoners = [CoderAgent(settings, answerer), ToolAgent(settings, answerer)]
    if settings.use_agentic_moe:
        router = ModelRouter(settings)
        models = [item.strip() for item in settings.agentic_moe_models.split(",") if item.strip()]
        exact_model = models[0] if models else router.model_for("exact")
        synthesis_model = models[1] if len(models) > 1 else router.model_for("synthesis")
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
                        "Use only the provided context. Return JSON with keys answer, evidences, rationale, claims. "
                        "Claims must be atomic statements with evidence_files and optional quotes. "
                        "For exact_match, answer with only the minimal value, label, option letter, date, or short phrase. "
                        "If any context file is plausibly relevant, give the best supported exact answer even when evidence is partial. "
                        "Use Not enough data to answer only when no context file is relevant at all."
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
                        "Return JSON with keys answer, evidences, rationale, claims. "
                        "Claims must be atomic statements with evidence_files and optional quotes. "
                        "Evidences must be file paths from the context. "
                        "If any context file is plausibly relevant, give a best-effort answer grounded in that context. "
                        "Use Not enough data to answer only when no context file is relevant at all."
                    ),
                ),
            ]
        )
    reasoners.append(FallbackReasoner(settings))
    return reasoners


def best_non_empty_candidate(state: AgentState, aggregator: CandidateAggregator) -> AnswerCandidate | None:
    usable = [
        candidate
        for candidate in state.candidates
        if candidate.answer and candidate.answer != "Not enough data to answer." and not candidate.diagnostics.get("failed")
    ]
    for candidate in sorted(usable, key=lambda item: item.confidence, reverse=True):
        evidences = [item for item in candidate.evidences if aggregator._valid_evidence_file(item, state)]
        if evidences:
            candidate.evidences = dedupe_keep_order(evidences)
            return candidate
        top_files = top_evidence_files(state.evidence_pool, aggregator.settings.max_files_for_question)
        top_files = [path for path in top_files if aggregator._valid_evidence_file(path, state)]
        if top_files:
            candidate.evidences = dedupe_keep_order(top_files)
            return candidate
    return None


def best_effort_static_answer(state: AgentState) -> str:
    text = best_evidence_text(state)
    question = state.question
    contract = detect_answer_contract(question)
    if not text:
        return "Not enough data to answer."
    if contract.format_name in {"integer", "number"} or question.answer_type.casefold() == "exact_match":
        number = guess_number_answer(question.question, text)
        if number:
            return number
    if contract.format_name == "single_option_letter":
        option = guess_option_answer(text)
        if option:
            return option
    snippet = best_matching_snippet(question.question, text)
    return normalize_answer(snippet[:220], question.answer_type.casefold() == "exact_match", question) or snippet[:220]


def best_evidence_text(state: AgentState) -> str:
    parts: list[str] = []
    budget = 3000 if state.question.answer_type.casefold() == "exact_match" else 2000
    for result in state.evidence_pool[:12]:
        if budget <= 0:
            break
        text = normalize_text(result.chunk.text)
        take = text[:budget]
        parts.append(f"[{result.chunk.file_path}]\n{take}")
        budget -= len(take)
    return "\n\n".join(parts)


def guess_number_answer(question: str, text: str) -> str:
    snippet = best_matching_snippet(question, text)
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", snippet or text)
    if not numbers:
        return ""
    cleaned = [number.replace(",", "") for number in numbers]
    q = question.casefold()
    if any(hint in q for hint in ["percent", "percentage", "tỷ lệ", "ti le", "%"]):
        for number in cleaned:
            try:
                value = float(number)
            except ValueError:
                continue
            if "." in number or 0 <= value <= 100:
                return number
    return cleaned[0]


def guess_option_answer(text: str) -> str:
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1).upper() if match else ""


def best_matching_snippet(question: str, text: str) -> str:
    query_tokens = {token for token in tokenize(question) if len(token) >= 3}
    candidates: list[tuple[int, bool, str]] = []
    for line in re.split(r"[\n.;]", text):
        snippet = normalize_text(line)
        if len(snippet) < 3:
            continue
        overlap = len(query_tokens & set(tokenize(snippet)))
        candidates.append((overlap, len(snippet) <= 220, snippet))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2] if candidates else normalize_text(text)[:220]
