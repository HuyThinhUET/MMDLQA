from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mmdlqa_core.openrouter import OpenRouterClient
from mmdlqa_core.prompting import secure_system_prompt
from mmdlqa_core.utils import json_dumps, normalize_text


Validator = Callable[[dict[str, Any]], "ValidationResult"]


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    errors: list[str]


@dataclass(slots=True)
class ValidatedJson:
    data: dict[str, Any]
    diagnostics: dict[str, Any]


def json_chat_validated(
    llm: OpenRouterClient,
    messages: list[dict[str, Any]],
    *,
    validator: Validator,
    schema_name: str,
    schema_hint: dict[str, Any],
    model: str | None = None,
    max_tokens: int = 1024,
    repair_max_tokens: int = 700,
) -> ValidatedJson:
    data = ensure_object(llm.json_chat(messages, model=model, max_tokens=max_tokens))
    first = validator(data)
    if first.ok:
        return ValidatedJson(data=data, diagnostics={"validated": True, "repaired": False})

    repair_data = ensure_object(
        repair_json_output(
            llm,
            schema_name=schema_name,
            schema_hint=schema_hint,
            invalid_output=data,
            errors=first.errors,
            original_messages=messages,
            model=model,
            max_tokens=repair_max_tokens,
        )
    )
    second = validator(repair_data)
    if second.ok:
        return ValidatedJson(
            data=repair_data,
            diagnostics={
                "validated": True,
                "repaired": True,
                "repair_errors": first.errors,
            },
        )
    raise ValueError(
        f"{schema_name} validation failed after repair: "
        + "; ".join([*first.errors, *second.errors])
    )


def repair_json_output(
    llm: OpenRouterClient,
    *,
    schema_name: str,
    schema_hint: dict[str, Any],
    invalid_output: dict[str, Any],
    errors: list[str],
    original_messages: list[dict[str, Any]],
    model: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    system = secure_system_prompt(
        "Repair an invalid JSON object for an agentic QA workflow. "
        "Return only one valid JSON object matching the requested schema. "
        "Do not add markdown, prose, or explanations."
    )
    payload = {
        "schema_name": schema_name,
        "schema_hint": schema_hint,
        "validation_errors": errors,
        "invalid_output": invalid_output,
        "original_task_preview": preview_messages(original_messages),
    }
    return llm.json_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json_dumps(payload)},
        ],
        model=model,
        max_tokens=max_tokens,
    )


def ensure_object(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    return {"_invalid_output": data}


def validate_planner_output(max_steps: int) -> Validator:
    def validate(data: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        steps = data.get("steps")
        if not isinstance(steps, list) or not steps:
            errors.append("steps must be a non-empty list")
            return ValidationResult(False, errors)
        if len(steps) > max_steps:
            errors.append(f"steps must contain at most {max_steps} items")
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"steps[{index}] must be an object")
                continue
            if not normalize_text(str(step.get("sentence", ""))):
                errors.append(f"steps[{index}].sentence is required")
            if step.get("depends_on") is not None and not isinstance(step.get("depends_on"), list):
                errors.append(f"steps[{index}].depends_on must be a list")
            if step.get("metadata") is not None and not isinstance(step.get("metadata"), dict):
                errors.append(f"steps[{index}].metadata must be an object")
        return ValidationResult(not errors, errors)

    return validate


def validate_rerank_output(candidate_count: int) -> Validator:
    def validate(data: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        selected = data.get("selected_indices")
        if not isinstance(selected, list):
            errors.append("selected_indices must be a list")
            return ValidationResult(False, errors)
        for index, value in enumerate(selected):
            if not isinstance(value, int):
                errors.append(f"selected_indices[{index}] must be an integer")
            elif value < 0 or value >= candidate_count:
                errors.append(f"selected_indices[{index}] is out of range")
        return ValidationResult(not errors, errors)

    return validate


def validate_answer_output(
    valid_files: set[str],
    *,
    require_claims: bool = True,
    allow_insufficient: bool = True,
) -> Validator:
    def validate(data: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        answer = data.get("answer")
        if not isinstance(answer, str) or not normalize_text(answer):
            errors.append("answer must be a non-empty string")
        if answer == "Not enough data to answer." and not allow_insufficient and valid_files:
            errors.append(
                "answer must be a best-effort answer because at least one valid context file is available"
            )
        evidences = data.get("evidences")
        if not isinstance(evidences, list):
            errors.append("evidences must be a list of file paths")
            evidences = []
        invalid = [str(item) for item in evidences if str(item) not in valid_files]
        if invalid:
            errors.append("evidences contain files not present in context: " + ", ".join(invalid[:5]))
        if answer != "Not enough data to answer." and not [item for item in evidences if str(item) in valid_files]:
            errors.append("supported answers must cite at least one valid evidence file")
        if require_claims and answer != "Not enough data to answer.":
            validate_claims(data.get("claims", data.get("claim_evidence", [])), valid_files, errors)
        return ValidationResult(not errors, errors)

    return validate


def validate_critic_output() -> Validator:
    def validate(data: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        if not isinstance(data.get("ok"), bool):
            errors.append("ok must be a boolean")
        if not isinstance(data.get("issues", []), list):
            errors.append("issues must be a list")
        if not isinstance(data.get("missing_queries", []), list):
            errors.append("missing_queries must be a list")
        return ValidationResult(not errors, errors)

    return validate


def validate_coder_plan_output() -> Validator:
    def validate(data: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        if not isinstance(data.get("should_run"), bool):
            errors.append("should_run must be a boolean")
        for key in ["task_type", "operation", "sheet_hint", "rationale"]:
            if data.get(key) is not None and not isinstance(data.get(key), str):
                errors.append(f"{key} must be a string")
        for key in ["target_files", "columns"]:
            if data.get(key) is not None and not isinstance(data.get(key), list):
                errors.append(f"{key} must be a list")
        return ValidationResult(not errors, errors)

    return validate


def validate_evidence_scan_output(candidate_count: int | set[int]) -> Validator:
    valid_indices = set(range(candidate_count)) if isinstance(candidate_count, int) else set(candidate_count)

    def validate(data: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        assessments = data.get("assessments")
        if not isinstance(assessments, list):
            errors.append("assessments must be a list")
            return ValidationResult(False, errors)
        allowed = {"direct", "partial", "irrelevant"}
        seen: set[int] = set()
        for index, item in enumerate(assessments):
            if not isinstance(item, dict):
                errors.append(f"assessments[{index}] must be an object")
                continue
            item_index = item.get("i")
            if not isinstance(item_index, int):
                errors.append(f"assessments[{index}].i must be an integer")
            elif item_index not in valid_indices:
                errors.append(f"assessments[{index}].i is out of range")
            else:
                seen.add(item_index)
            relevance = str(item.get("relevance", "")).strip().casefold()
            if relevance not in allowed:
                errors.append(f"assessments[{index}].relevance must be direct, partial, or irrelevant")
            snippets = item.get("useful_snippets", [])
            if snippets is not None and not isinstance(snippets, list):
                errors.append(f"assessments[{index}].useful_snippets must be a list")
            confidence = item.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)):
                errors.append(f"assessments[{index}].confidence must be a number")
        if not seen:
            errors.append("at least one candidate assessment is required")
        return ValidationResult(not errors, errors)

    return validate


def validate_claims(value: Any, valid_files: set[str], errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append("claims must be a non-empty list for supported answers")
        return
    for index, claim in enumerate(value):
        if not isinstance(claim, dict):
            errors.append(f"claims[{index}] must be an object")
            continue
        if not normalize_text(str(claim.get("claim", claim.get("statement", "")))):
            errors.append(f"claims[{index}].claim is required")
        files = claim.get("evidence_files", claim.get("evidences", claim.get("files", [])))
        if not isinstance(files, list) or not files:
            errors.append(f"claims[{index}].evidence_files must be a non-empty list")
            continue
        invalid = [str(item) for item in files if str(item) not in valid_files]
        if invalid:
            errors.append(f"claims[{index}] cites files not present in context: " + ", ".join(invalid[:5]))


def preview_messages(messages: list[dict[str, Any]], limit: int = 6000) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    used = 0
    for message in messages:
        content = preview_content(message.get("content"))
        remaining = max(0, limit - used)
        if remaining <= 0:
            break
        content = content[:remaining]
        used += len(content)
        out.append({"role": message.get("role", ""), "content": content})
    return out


def preview_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "image_url":
                parts.append("[image_url omitted]")
            else:
                parts.append(str({k: v for k, v in item.items() if k != "image_url"}))
        return "\n".join(parts)
    return str(content)
