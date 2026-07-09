from __future__ import annotations

from mmdlqa_core.config import Settings
from mmdlqa_core.contracts import AgentState, AnswerCandidate, CriticReport, RagQuery, ReasoningStep
from mmdlqa_core.schema import AnswerResult, Question, RetrievedChunk
from mmdlqa_retrieval.rag import SentenceRAG

from .answering import Answerer


class RuleBasedCoordinator:
    def plan(self, question: Question) -> list[ReasoningStep]:
        return [
            ReasoningStep(
                step_id=f"{question.qid}:q0",
                sentence=question.question,
                purpose="answer_question",
                metadata={"source": "original_question"},
            )
        ]


class EvidenceFormatCritic:
    def review(self, state: AgentState, candidate: AnswerCandidate) -> CriticReport:
        issues: list[str] = []
        available_files = {result.chunk.file_path for result in state.evidence_pool}
        invalid_evidences = [e for e in candidate.evidences if e not in available_files]
        if not candidate.answer:
            issues.append("empty_answer")
        if invalid_evidences:
            issues.append("evidence_not_in_retrieved_pool: " + ", ".join(invalid_evidences))
        if not state.evidence_pool and candidate.answer != "Not enough data to answer.":
            issues.append("answered_without_evidence")

        missing_queries = []
        if not state.evidence_pool:
            missing_queries.append(state.question.question)

        return CriticReport(
            reviewer="evidence_format_critic",
            ok=not issues,
            issues=issues,
            missing_queries=missing_queries,
        )


class AgenticAnswerer:
    def __init__(
        self,
        settings: Settings,
        rag: SentenceRAG,
        *,
        coordinator: RuleBasedCoordinator | None = None,
        critic: EvidenceFormatCritic | None = None,
        reasoner: Answerer | None = None,
    ):
        self.settings = settings
        self.rag = rag
        self.coordinator = coordinator or RuleBasedCoordinator()
        self.critic = critic or EvidenceFormatCritic()
        self.reasoner = reasoner or Answerer(settings)

    def answer(self, question: Question) -> AnswerResult:
        state = AgentState(question=question)
        state.steps = self.coordinator.plan(question)
        for step in state.steps:
            rag_query = RagQuery(
                sentence=step.sentence,
                purpose=step.purpose,
                step_id=step.step_id,
                metadata={"answer_type": question.answer_type},
            )
            state.rag_queries.append(rag_query)
            retrieved = self.rag.search_sentence(rag_query)
            state.evidence_pool = merge_retrieved(state.evidence_pool, retrieved)

        result = self.reasoner.answer(question, state.evidence_pool)
        candidate = AnswerCandidate(
            source=str(result.diagnostics.get("method", "reasoner")),
            answer=result.answer,
            evidences=result.evidences,
            confidence=1.0 if result.answer != "Not enough data to answer." else 0.0,
            diagnostics=result.diagnostics,
        )
        state.candidates.append(candidate)
        report = self.critic.review(state, candidate)
        state.critic_reports.append(report)
        state.final_answer = result

        result.diagnostics = {
            **result.diagnostics,
            "agentic": summarize_state(state),
        }
        return result


def merge_retrieved(existing: list[RetrievedChunk], new: list[RetrievedChunk]) -> list[RetrievedChunk]:
    by_id = {result.chunk.chunk_id: result for result in existing}
    for result in new:
        old = by_id.get(result.chunk.chunk_id)
        if old is None or result.score > old.score:
            by_id[result.chunk.chunk_id] = result
    return sorted(by_id.values(), key=lambda r: r.score, reverse=True)


def summarize_state(state: AgentState) -> dict:
    return {
        "steps": [
            {
                "step_id": step.step_id,
                "purpose": step.purpose,
                "sentence": step.sentence,
                "depends_on": step.depends_on,
            }
            for step in state.steps
        ],
        "rag_queries": [
            {
                "step_id": query.step_id,
                "purpose": query.purpose,
                "sentence": query.sentence,
                "top_k": query.top_k,
            }
            for query in state.rag_queries
        ],
        "evidence_count": len(state.evidence_pool),
        "evidence_files": sorted({result.chunk.file_path for result in state.evidence_pool}),
        "critic_reports": [
            {
                "reviewer": report.reviewer,
                "ok": report.ok,
                "issues": report.issues,
                "missing_queries": report.missing_queries,
            }
            for report in state.critic_reports
        ],
    }
