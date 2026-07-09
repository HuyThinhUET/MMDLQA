from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mmdlqa_core.config import Settings
from mmdlqa_core.contracts import AgentState, AnswerCandidate, ToolCallRecord, ToolPlan
from mmdlqa_core.metrics import BudgetExceededError
from mmdlqa_core.model_router import ModelRouter
from mmdlqa_core.openrouter import OpenRouterClient
from mmdlqa_core.prompting import secure_system_prompt, untrusted_data_notice
from mmdlqa_core.schema import AnswerResult, RetrievedChunk
from mmdlqa_core.utils import json_dumps, normalize_text

from .answering import Answerer
from .evidence import ledger_from_result
from .structured import json_chat_validated, validate_coder_plan_output


class ToolAgent:
    def __init__(self, settings: Settings, answerer: Answerer):
        self.settings = settings
        self.answerer = answerer

    def run(self, state: AgentState) -> list[AnswerCandidate]:
        if not self.settings.use_agentic_tools:
            return []
        candidates: list[AnswerCandidate] = []
        vision = self.answerer.maybe_answer_image_group(state.question, state.evidence_pool)
        if vision:
            candidates.append(candidate_from_result("tool_agent_vision", vision, state.evidence_pool, confidence=0.9))
            state.tool_calls.append(
                ToolCallRecord(
                    tool_name="vision_group",
                    input={"question": state.question.question},
                    output={"answer": vision.answer, "evidences": vision.evidences},
                )
            )

        deterministic = self.answerer.tools.try_tool_answer(state.question, state.evidence_pool)
        if deterministic:
            candidates.append(
                candidate_from_result("tool_agent_deterministic", deterministic, state.evidence_pool, confidence=0.88)
            )
            state.tool_calls.append(
                ToolCallRecord(
                    tool_name=str(deterministic.diagnostics.get("method", "deterministic_tool")),
                    input={"question": state.question.question},
                    output={"answer": deterministic.answer, "evidences": deterministic.evidences},
                )
            )
        return candidates


class CoderAgent:
    def __init__(self, settings: Settings, answerer: Answerer):
        self.settings = settings
        self.answerer = answerer
        self.llm = OpenRouterClient(settings)
        self.models = ModelRouter(settings)

    def run(self, state: AgentState) -> list[AnswerCandidate]:
        if not self.settings.use_agentic_coder or not needs_coder_task(state):
            return []
        try:
            plan = self.plan(state)
        except BudgetExceededError:
            raise
        except Exception as exc:
            plan = rule_coder_plan(state)
            plan.diagnostics["planner_error"] = repr(exc)
        state.tool_calls.append(
            ToolCallRecord(
                tool_name="coder_plan",
                input={"question": state.question.question},
                output=asdict(plan),
                ok=plan.should_run,
            )
        )
        if not plan.should_run:
            return []

        result = self.answerer.tools.try_coder_answer(state.question, state.evidence_pool)
        if not result:
            state.tool_calls.append(
                ToolCallRecord(
                    tool_name="coder_executor",
                    input=asdict(plan),
                    output={"answer": ""},
                    ok=False,
                    error="no_whitelisted_tool_matched",
                )
            )
            return []
        state.tool_calls.append(
            ToolCallRecord(
                tool_name=str(result.diagnostics.get("method", "coder_executor")),
                input=asdict(plan),
                output={"answer": result.answer, "evidences": result.evidences},
            )
        )
        return [candidate_from_result("coder_agent", result, state.evidence_pool, confidence=0.96)]

    def plan(self, state: AgentState) -> ToolPlan:
        if not (self.settings.use_coder_planner and self.llm.available):
            return rule_coder_plan(state)
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
            "steps": [
                {"step_id": step.step_id, "purpose": step.purpose, "sentence": step.sentence}
                for step in state.steps
            ],
            "context": context,
            "prompt_security": untrusted_data_notice(),
            "executor_policy": {
                "no_arbitrary_code_execution": True,
                "allowed_outputs": ["plan_only"],
                "whitelisted_executor": "DeterministicToolbox.try_coder_answer",
            },
        }
        schema_hint = {
            "should_run": "boolean",
            "task_type": "table_calculation|sql_lookup|spreadsheet_count|correlation|other",
            "operation": "short operation name",
            "target_files": ["file paths from context"],
            "columns": ["column names when relevant"],
            "sheet_hint": "sheet hint when relevant",
            "rationale": "short reason for the plan",
        }
        validated = json_chat_validated(
            self.llm,
            [
                {
                    "role": "system",
                    "content": secure_system_prompt(
                        "You are a safe coder planner for data-lake QA. "
                        "Return JSON for a calculation/tool plan only. Do not output executable code. "
                        "The executor is whitelisted and will decide whether a supported deterministic tool exists."
                    ),
                },
                {"role": "user", "content": json_dumps(payload)},
            ],
            validator=validate_coder_plan_output(),
            schema_name="coder_plan",
            schema_hint=schema_hint,
            model=self.models.model_for("coder"),
            max_tokens=700,
            repair_max_tokens=500,
        )
        data = validated.data
        return ToolPlan(
            agent="coder_agent",
            should_run=bool(data.get("should_run")),
            task_type=normalize_text(str(data.get("task_type", ""))),
            operation=normalize_text(str(data.get("operation", ""))),
            target_files=[str(item) for item in data.get("target_files", []) if normalize_text(str(item))],
            columns=[str(item) for item in data.get("columns", []) if normalize_text(str(item))],
            sheet_hint=normalize_text(str(data.get("sheet_hint", ""))),
            rationale=normalize_text(str(data.get("rationale", ""))),
            diagnostics=validated.diagnostics,
        )


def candidate_from_result(
    source: str,
    result: AnswerResult,
    retrieved: list[RetrievedChunk],
    *,
    confidence: float,
) -> AnswerCandidate:
    return AnswerCandidate(
        source=source,
        answer=result.answer,
        evidences=result.evidences,
        claim_evidence=ledger_from_result(source, result, retrieved, supported=True),
        confidence=confidence,
        diagnostics=result.diagnostics,
    )


def needs_coder_task(state: AgentState) -> bool:
    text = " ".join(
        [
            state.question.question,
            state.question.answer_type,
            *[step.purpose + " " + step.sentence for step in state.steps],
        ]
    ).casefold()
    hints = [
        "table_calculation",
        "calculate",
        "correlation",
        "coefficient",
        "average",
        "sum",
        "count",
        "how many",
        "significant genes",
        "sql",
        "csv",
        "xlsx",
        "sheet",
        "workbook",
        "trung binh",
        "bao nhieu",
    ]
    return any(hint in text for hint in hints)


def rule_coder_plan(state: AgentState) -> ToolPlan:
    files = []
    for result in state.evidence_pool:
        suffix = result.chunk.file_path.rsplit(".", 1)[-1].casefold()
        if suffix in {"csv", "tsv", "xlsx", "xls", "sql"}:
            files.append(result.chunk.file_path)
    operation = infer_operation(state)
    return ToolPlan(
        agent="coder_agent",
        should_run=needs_coder_task(state),
        task_type=infer_task_type(state),
        operation=operation,
        target_files=list(dict.fromkeys(files))[:8],
        rationale="rule_based_coder_plan",
        diagnostics={"planner": "rules"},
    )


def infer_task_type(state: AgentState) -> str:
    text = state.question.question.casefold()
    if "correlation" in text or "coefficient" in text:
        return "correlation"
    if "sql" in text or any(result.chunk.file_path.casefold().endswith(".sql") for result in state.evidence_pool):
        return "sql_lookup"
    if "significant genes" in text or "sheet" in text or "workbook" in text:
        return "spreadsheet_count"
    return "table_calculation"


def infer_operation(state: AgentState) -> str:
    text = state.question.question.casefold()
    if "correlation" in text or "coefficient" in text:
        return "pearson_correlation"
    if "average" in text or "trung binh" in text:
        return "average"
    if "significant genes" in text:
        return "count_significant_genes"
    if "how many" in text or "count" in text or "bao nhieu" in text:
        return "count"
    return "calculate"
