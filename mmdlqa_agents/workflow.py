from __future__ import annotations

from mmdlqa_core.config import Settings
from mmdlqa_core.contracts import AgentState, AnswerCandidate, RagQuery
from mmdlqa_core.metrics import BudgetExceededError, current_tracker
from mmdlqa_core.model_router import ModelRouter
from mmdlqa_core.schema import AnswerResult, Question, RetrievedChunk
from mmdlqa_core.utils import dedupe_keep_order, normalize_text
from mmdlqa_retrieval.hybrid import top_evidence_files
from mmdlqa_retrieval.rag import SentenceRAG

from .answering import Answerer
from .critic import EvidenceCritic
from .evidence import ledger_to_dicts
from .evidence_scanner import EvidenceScanner
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
        scanner: EvidenceScanner | None = None,
        reasoner: Answerer | None = None,
        moe: MultiExpertReasoner | None = None,
        aggregator: CandidateAggregator | None = None,
    ):
        self.settings = settings
        self.models = ModelRouter(settings)
        self.rag = rag
        self.reasoner = reasoner or Answerer(settings)
        self.planner = planner or QuestionPlanner(settings)
        self.critic = critic or EvidenceCritic(settings)
        self.scanner = scanner or EvidenceScanner(settings)
        self.moe = moe or MultiExpertReasoner(settings, self.reasoner)
        self.aggregator = aggregator or CandidateAggregator(settings)

    def answer(self, question: Question) -> AnswerResult:
        state = AgentState(question=question)
        tracker = current_tracker()
        try:
            if tracker:
                with tracker.stage("agentic_planning"):
                    state.steps = self.planner.plan(question)
                with tracker.stage("agentic_planned_retrieval"):
                    self.retrieve_planned_steps(state)
            else:
                state.steps = self.planner.plan(question)
                self.retrieve_planned_steps(state)
        except BudgetExceededError as exc:
            state.diagnostics["limit_stop"] = str(exc)

        selected = AnswerCandidate(source="empty", answer="Not enough data to answer.")
        rounds = max(1, self.settings.agentic_max_rounds)
        for round_idx in range(rounds):
            try:
                if tracker:
                    tracker.check_limits(f"before_reasoning_round_{round_idx}")
                    with tracker.stage("agentic_reasoning", {"round": round_idx}):
                        new_candidates = self.moe.run(state)
                    state.candidates.extend(new_candidates)
                    selected = self.aggregator.choose(state)
                    with tracker.stage("agentic_critic", {"round": round_idx}):
                        report = self.critic.review(state, selected)
                else:
                    new_candidates = self.moe.run(state)
                    state.candidates.extend(new_candidates)
                    selected = self.aggregator.choose(state)
                    report = self.critic.review(state, selected)
            except BudgetExceededError as exc:
                state.diagnostics["limit_stop"] = str(exc)
                break
            state.critic_reports.append(report)
            if report.ok or round_idx >= rounds - 1 or not report.missing_queries:
                break
            state.diagnostics["retried_missing_evidence"] = True
            try:
                if tracker:
                    with tracker.stage("agentic_critic_followup_retrieval", {"round": round_idx}):
                        self.retrieve_missing_queries(state, report.missing_queries, round_idx)
                else:
                    self.retrieve_missing_queries(state, report.missing_queries, round_idx)
            except BudgetExceededError as exc:
                state.diagnostics["limit_stop"] = str(exc)
                break

        selected = self.aggregator.ensure_best_effort(state, selected)
        if selected.source == "best_effort_static" and all(c.source != selected.source for c in state.candidates):
            state.candidates.append(selected)
        final_static_report = self.critic.static_review(state, selected)
        final_static_report.diagnostics["final_best_effort_review"] = True
        state.critic_reports.append(final_static_report)

        result = result_from_candidate(question, selected, state, self.settings)
        state.evidence_ledger = selected.claim_evidence
        state.final_answer = result

        result.diagnostics = {
            **result.diagnostics,
            "model_routing": self.models.snapshot(),
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
        tracker = current_tracker()
        if tracker:
            tracker.check_limits("before_rag_query")
        if (
            self.settings.max_question_rag_queries > 0
            and len(state.rag_queries) >= self.settings.max_question_rag_queries
        ):
            state.diagnostics["rag_query_limit_reached"] = True
            if tracker:
                tracker.note_limit("rag_query_limit_reached")
            return
        if query.top_k is None and (should_expand_retrieval(query) or self.settings.use_evidence_scanner):
            query.top_k = max(
                self.settings.retrieve_top_k,
                self.settings.max_files_for_question * 3,
                self.settings.evidence_scan_max_files * self.settings.evidence_scan_chunks_per_file,
                30,
            )
        state.rag_queries.append(query)
        retrieved = self.rag.search_sentence(query)
        if self.scanner.enabled:
            tracker = current_tracker()
            if tracker:
                with tracker.stage("agentic_evidence_scan", {"step_id": query.step_id}):
                    retrieved, scan_diag = self.scanner.scan(state.question, query, retrieved)
            else:
                retrieved, scan_diag = self.scanner.scan(state.question, query, retrieved)
            state.diagnostics.setdefault("evidence_scans", []).append(scan_diag)
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
    answer = candidate.answer or "Not enough data to answer."
    if settings.force_best_effort_answer and answer == "Not enough data to answer." and evidences:
        answer = fallback_answer_from_valid_evidence(question, state, evidences)
    return AnswerResult(
        qid=question.qid,
        answer=answer,
        evidences=dedupe_keep_order(evidences[: settings.max_files_for_question]),
        diagnostics={
            "method": "agentic",
            "selected_candidate": candidate.source,
            "candidate_confidence": candidate.confidence,
            "candidate_diagnostics": candidate.diagnostics,
            "evidence_ledger": ledger_to_dicts(candidate.claim_evidence),
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


def fallback_answer_from_valid_evidence(question: Question, state: AgentState, evidences: list[str]) -> str:
    for result in state.evidence_pool:
        if result.chunk.file_path not in evidences:
            continue
        text = normalize_text(result.chunk.text)
        if text:
            line = best_line_for_question(question.question, text)
            if question.answer_type.casefold() == "exact_match":
                import re

                numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", line)
                if numbers:
                    return numbers[0].replace(",", "")
            return line[:220] or text[:220]
    return evidences[0]


def best_line_for_question(question: str, text: str) -> str:
    import re

    q_tokens = {token.casefold() for token in re.findall(r"\w+", question) if len(token) >= 3}
    best = ""
    best_score = -1
    for line in re.split(r"[\n.;]", text):
        line = normalize_text(line)
        if not line:
            continue
        tokens = {token.casefold() for token in re.findall(r"\w+", line)}
        score = len(q_tokens & tokens)
        if score > best_score:
            best = line
            best_score = score
    return best or normalize_text(text)


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
        "evidence_ledger": ledger_to_dicts(state.evidence_ledger),
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
                "claim_evidence": ledger_to_dicts(candidate.claim_evidence),
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
