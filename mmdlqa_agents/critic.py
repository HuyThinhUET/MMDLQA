from __future__ import annotations

from dataclasses import asdict

from mmdlqa_core.config import Settings
from mmdlqa_core.contracts import AgentState, AnswerCandidate, CriticReport
from mmdlqa_core.openrouter import OpenRouterClient
from mmdlqa_core.utils import json_dumps, normalize_text


class EvidenceCritic:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = OpenRouterClient(settings)

    def review(self, state: AgentState, candidate: AnswerCandidate) -> CriticReport:
        static = self._static_review(state, candidate)
        if not (self.settings.use_agentic_critic and self.llm.available and state.evidence_pool):
            return static
        try:
            llm_report = self._llm_review(state, candidate)
        except Exception as exc:
            static.diagnostics["llm_critic_error"] = repr(exc)
            return static
        return merge_reports(static, llm_report)

    def _static_review(self, state: AgentState, candidate: AnswerCandidate) -> CriticReport:
        issues: list[str] = []
        missing_queries: list[str] = []
        invalid_evidences = [e for e in candidate.evidences if not self._valid_evidence_file(e, state)]
        if not candidate.answer:
            issues.append("empty_answer")
        if invalid_evidences:
            issues.append("evidence_not_in_retrieved_pool: " + ", ".join(invalid_evidences))
        if candidate.answer != "Not enough data to answer." and not candidate.evidences:
            issues.append("answer_without_evidence")
            missing_queries.append(f"Find source evidence for this answer: {state.question.question}")
        if not state.evidence_pool and candidate.answer != "Not enough data to answer.":
            issues.append("answered_without_retrieved_context")
            missing_queries.append(state.question.question)
        if state.question.answer_type.casefold() == "exact_match":
            token_count = len(candidate.answer.split())
            if token_count > 24:
                issues.append("exact_match_answer_too_long")

        if candidate.answer == "Not enough data to answer." and not state.diagnostics.get("retried_missing_evidence"):
            missing_queries.append(f"Find any missing evidence needed to answer: {state.question.question}")

        return CriticReport(
            reviewer="static_evidence_critic",
            ok=not issues,
            issues=issues,
            missing_queries=dedupe_queries(missing_queries),
            diagnostics={"candidate_source": candidate.source},
        )

    def _valid_evidence_file(self, path: str, state: AgentState) -> bool:
        if path in {result.chunk.file_path for result in state.evidence_pool}:
            return True
        return (self.settings.raw_dir / path).exists()

    def _llm_review(self, state: AgentState, candidate: AnswerCandidate) -> CriticReport:
        context = [
            {
                "file": result.chunk.file_path,
                "modality": result.chunk.modality,
                "score": round(result.score, 4),
                "text_preview": result.chunk.text[:900],
            }
            for result in state.evidence_pool[: self.settings.rerank_top_k]
        ]
        payload = {
            "question": asdict(state.question),
            "candidate": {
                "source": candidate.source,
                "answer": candidate.answer,
                "evidences": candidate.evidences,
                "rationale": candidate.rationale,
            },
            "steps": [
                {"step_id": step.step_id, "purpose": step.purpose, "sentence": step.sentence}
                for step in state.steps
            ],
            "context": context,
        }
        data = self.llm.json_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict verifier for data-lake QA. "
                        "Check whether the candidate answer is supported by the provided context only. "
                        "Return JSON with keys ok (boolean), issues (array of short strings), "
                        "missing_queries (array of standalone RAG query sentences). "
                        "Use missing_queries only when another retrieval pass could fix the answer."
                    ),
                },
                {"role": "user", "content": json_dumps(payload)},
            ],
            max_tokens=700,
        )
        issues = data.get("issues", [])
        missing_queries = data.get("missing_queries", [])
        if not isinstance(issues, list):
            issues = []
        if not isinstance(missing_queries, list):
            missing_queries = []
        ok = bool(data.get("ok", not issues))
        return CriticReport(
            reviewer="llm_evidence_critic",
            ok=ok,
            issues=[normalize_text(str(item)) for item in issues if normalize_text(str(item))],
            missing_queries=dedupe_queries(str(item) for item in missing_queries),
            diagnostics={"model": self.settings.openrouter_model},
        )


def merge_reports(static: CriticReport, llm_report: CriticReport) -> CriticReport:
    issues = dedupe_queries([*static.issues, *llm_report.issues])
    missing_queries = dedupe_queries([*static.missing_queries, *llm_report.missing_queries])
    return CriticReport(
        reviewer=f"{static.reviewer}+{llm_report.reviewer}",
        ok=static.ok and llm_report.ok,
        issues=issues,
        missing_queries=missing_queries,
        diagnostics={**static.diagnostics, **llm_report.diagnostics},
    )


def dedupe_queries(values) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = normalize_text(str(value))
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out
