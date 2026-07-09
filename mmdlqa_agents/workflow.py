from __future__ import annotations

from mmdlqa_core.config import Settings
from mmdlqa_core.contracts import AgentState, AnswerCandidate, RagQuery
from mmdlqa_core.schema import AnswerResult, Question, RetrievedChunk
from mmdlqa_core.utils import dedupe_keep_order
from mmdlqa_retrieval.hybrid import top_evidence_files
from mmdlqa_retrieval.rag import SentenceRAG

from .answering import Answerer
from .critic import EvidenceCritic
from .planner import QuestionPlanner
from .reasoners import CandidateAggregator, MultiExpertReasoner


class AgenticAnswerer:
    def __init__(
        self,
        settings: Settings,
        rag: SentenceRAG,
        *,
        planner: QuestionPlanner | None = None,
        critic: EvidenceCritic | None = None,
        reasoner: Answerer | None = None,
        moe: MultiExpertReasoner | None = None,
        aggregator: CandidateAggregator | None = None,
    ):
        self.settings = settings
        self.rag = rag
        self.reasoner = reasoner or Answerer(settings)
        self.planner = planner or QuestionPlanner(settings)
        self.critic = critic or EvidenceCritic(settings)
        self.moe = moe or MultiExpertReasoner(settings, self.reasoner)
        self.aggregator = aggregator or CandidateAggregator(settings)

    def answer(self, question: Question) -> AnswerResult:
        state = AgentState(question=question)
        state.steps = self.planner.plan(question)
        self.retrieve_planned_steps(state)

        selected = AnswerCandidate(source="empty", answer="Not enough data to answer.")
        rounds = max(1, self.settings.agentic_max_rounds)
        for round_idx in range(rounds):
            new_candidates = self.moe.run(state)
            state.candidates.extend(new_candidates)
            selected = self.aggregator.choose(state)
            report = self.critic.review(state, selected)
            state.critic_reports.append(report)
            if report.ok or round_idx >= rounds - 1 or not report.missing_queries:
                break
            state.diagnostics["retried_missing_evidence"] = True
            self.retrieve_missing_queries(state, report.missing_queries, round_idx)

        result = result_from_candidate(question, selected, state, self.settings)
        state.final_answer = result

        result.diagnostics = {
            **result.diagnostics,
            "agentic": summarize_state(state),
        }
        return result

    def retrieve_planned_steps(self, state: AgentState) -> None:
        for step in state.steps:
            query = RagQuery(
                sentence=step.sentence,
                purpose=step.purpose,
                step_id=step.step_id,
                metadata={"answer_type": state.question.answer_type, **step.metadata},
            )
            self.retrieve_query(state, query)

    def retrieve_missing_queries(self, state: AgentState, queries: list[str], round_idx: int) -> None:
        for i, sentence in enumerate(queries[: self.settings.agentic_max_steps]):
            query = RagQuery(
                sentence=sentence,
                purpose="critic_followup",
                step_id=f"{state.question.qid}:critic{round_idx}:{i}",
                metadata={"answer_type": state.question.answer_type, "source": "critic"},
            )
            self.retrieve_query(state, query)

    def retrieve_query(self, state: AgentState, query: RagQuery) -> None:
        if query.top_k is None and should_expand_retrieval(query):
            query.top_k = max(self.settings.retrieve_top_k, self.settings.max_files_for_question * 3, 30)
        state.rag_queries.append(query)
        retrieved = self.rag.search_sentence(query)
        state.evidence_pool = merge_retrieved(state.evidence_pool, retrieved)


def merge_retrieved(existing: list[RetrievedChunk], new: list[RetrievedChunk]) -> list[RetrievedChunk]:
    by_id = {result.chunk.chunk_id: result for result in existing}
    for result in new:
        old = by_id.get(result.chunk.chunk_id)
        if old is None or result.score > old.score:
            by_id[result.chunk.chunk_id] = result
    return sorted(by_id.values(), key=lambda r: r.score, reverse=True)


def result_from_candidate(
    question: Question,
    candidate: AnswerCandidate,
    state: AgentState,
    settings: Settings,
) -> AnswerResult:
    evidences = [item for item in candidate.evidences if is_valid_evidence_file(item, state, settings)]
    if not evidences and candidate.answer and candidate.answer != "Not enough data to answer.":
        evidences = top_evidence_files(state.evidence_pool, settings.max_files_for_question)
    return AnswerResult(
        qid=question.qid,
        answer=candidate.answer or "Not enough data to answer.",
        evidences=dedupe_keep_order(evidences[: settings.max_files_for_question]),
        diagnostics={
            "method": "agentic",
            "selected_candidate": candidate.source,
            "candidate_confidence": candidate.confidence,
            "candidate_diagnostics": candidate.diagnostics,
        },
    )


def should_expand_retrieval(query: RagQuery) -> bool:
    sentence = query.sentence.casefold()
    media_or_folder = any(
        hint in sentence
        for hint in [
            "image",
            "images",
            "ảnh",
            "number_image",
            "folder",
            "audio",
            "video",
            "jpg",
            "png",
            "m4a",
            "*",
        ]
    )
    return query.purpose in {"source_retrieval", "image_understanding", "audio_understanding"} or media_or_folder


def is_valid_evidence_file(path: str, state: AgentState, settings: Settings) -> bool:
    if path in {result.chunk.file_path for result in state.evidence_pool}:
        return True
    return (settings.raw_dir / path).exists()


def summarize_state(state: AgentState) -> dict:
    return {
        "steps": [
            {
                "step_id": step.step_id,
                "purpose": step.purpose,
                "sentence": step.sentence,
                "depends_on": step.depends_on,
                "metadata": step.metadata,
            }
            for step in state.steps
        ],
        "rag_queries": [
            {
                "step_id": query.step_id,
                "purpose": query.purpose,
                "sentence": query.sentence,
                "top_k": query.top_k,
                "metadata": query.metadata,
            }
            for query in state.rag_queries
        ],
        "evidence_count": len(state.evidence_pool),
        "evidence_files": sorted({result.chunk.file_path for result in state.evidence_pool}),
        "tool_calls": [
            {
                "tool_name": call.tool_name,
                "input": call.input,
                "output": call.output,
                "ok": call.ok,
                "error": call.error,
            }
            for call in state.tool_calls
        ],
        "candidates": [
            {
                "source": candidate.source,
                "answer": candidate.answer,
                "evidences": candidate.evidences,
                "confidence": candidate.confidence,
                "rationale": candidate.rationale,
                "diagnostics": candidate.diagnostics,
            }
            for candidate in state.candidates
        ],
        "critic_reports": [
            {
                "reviewer": report.reviewer,
                "ok": report.ok,
                "issues": report.issues,
                "missing_queries": report.missing_queries,
                "diagnostics": report.diagnostics,
            }
            for report in state.critic_reports
        ],
        "diagnostics": state.diagnostics,
    }
