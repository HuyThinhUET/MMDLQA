from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from mmdlqa_core.config import Settings
from mmdlqa_core.openrouter import OpenRouterClient, image_part_from_path
from mmdlqa_core.schema import AnswerResult, Question, RetrievedChunk
from mmdlqa_core.utils import dedupe_keep_order, json_dumps, normalize_text
from mmdlqa_retrieval.hybrid import top_evidence_files

from .tools import DeterministicToolbox


class Answerer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = OpenRouterClient(settings)
        self.tools = DeterministicToolbox(settings)

    def answer(self, question: Question, retrieved: list[RetrievedChunk]) -> AnswerResult:
        vision_group = self.maybe_answer_image_group(question, retrieved)
        if vision_group:
            return vision_group

        deterministic = self.try_deterministic(question, retrieved)
        if deterministic:
            return deterministic

        if not retrieved:
            return AnswerResult(
                qid=question.qid,
                answer="Not enough data to answer.",
                evidences=[],
                diagnostics={"reason": "no_retrieved_chunks"},
            )

        if self.llm.available:
            try:
                return self.answer_with_llm(question, retrieved)
            except Exception as exc:
                fallback = self.fallback_answer(question, retrieved)
                fallback.diagnostics["llm_error"] = repr(exc)
                return fallback
        return self.fallback_answer(question, retrieved)

    def try_deterministic(self, question: Question, retrieved: list[RetrievedChunk]) -> AnswerResult | None:
        return self.tools.try_answer(question, retrieved)

    def maybe_answer_image_group(
        self, question: Question, retrieved: list[RetrievedChunk]
    ) -> AnswerResult | None:
        q = question.question.casefold()
        if not any(hint in q for hint in ["image", "images", "ảnh", "digit", "jpg", "png"]):
            return None
        image_paths: list[Path] = []
        file_paths: list[str] = []
        seen: set[str] = set()
        for result in retrieved:
            if result.chunk.modality != "image" or result.chunk.file_path in seen:
                continue
            seen.add(result.chunk.file_path)
            path = Path(str(result.chunk.metadata.get("abs_path", "")))
            if path.exists():
                image_paths.append(path)
                file_paths.append(result.chunk.file_path)
        return self.answer_image_group_with_vision(question, image_paths, file_paths)

    def answer_with_llm(self, question: Question, retrieved: list[RetrievedChunk]) -> AnswerResult:
        narrowed = self.rerank(question, retrieved)
        context = self.build_context(narrowed)
        exact = question.answer_type.casefold() == "exact_match"
        system = (
            "You answer questions using only the provided data-lake context. "
            "Return JSON with keys answer and evidences. Evidences must be a JSON array of file paths from the context. "
            "If the context is insufficient, answer exactly: Not enough data to answer. "
            "For exact_match questions, keep answer minimal: number, label, option letter, date, or short phrase only. "
            "Do not invent external facts."
        )
        user = {
            "question_id": question.qid,
            "question": question.question,
            "answer_type": question.answer_type or "unknown",
            "exact_match_style": exact,
            "context": context,
        }
        data = self.llm.json_chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json_dumps(user)},
            ],
            max_tokens=1200,
        )
        answer = normalize_answer(str(data.get("answer", "")), exact)
        evidences = data.get("evidences", [])
        if not isinstance(evidences, list):
            evidences = []
        valid_files = {r.chunk.file_path for r in narrowed}
        evidences = [str(e) for e in evidences if str(e) in valid_files]
        if not evidences and answer != "Not enough data to answer.":
            evidences = top_evidence_files(narrowed, self.settings.max_files_for_question)
        return AnswerResult(
            qid=question.qid,
            answer=answer or "Not enough data to answer.",
            evidences=dedupe_keep_order(evidences),
            diagnostics={"method": "llm", "context_chunks": len(narrowed)},
        )

    def rerank(self, question: Question, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        candidates = retrieved[: self.settings.retrieve_top_k]
        if not (self.settings.use_llm_rerank and self.llm.available and candidates):
            return candidates[: self.settings.rerank_top_k]
        items = [
            {
                "i": i,
                "file": r.chunk.file_path,
                "modality": r.chunk.modality,
                "text_preview": r.chunk.text[:900],
            }
            for i, r in enumerate(candidates)
        ]
        try:
            data = self.llm.json_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Rank data-lake chunks by usefulness for answering the question. "
                            "Return JSON: {\"selected_indices\":[...]} with the best chunks first."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json_dumps({"question": question.question, "candidates": items}),
                    },
                ],
                max_tokens=500,
            )
            selected = data.get("selected_indices", [])
            ranked = []
            for idx in selected:
                if isinstance(idx, int) and 0 <= idx < len(candidates):
                    ranked.append(candidates[idx])
            for cand in candidates:
                if cand not in ranked:
                    ranked.append(cand)
            return ranked[: self.settings.rerank_top_k]
        except Exception:
            return candidates[: self.settings.rerank_top_k]

    def build_context(self, retrieved: list[RetrievedChunk]) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        budget = self.settings.max_context_chars
        for result in retrieved:
            if budget <= 0:
                break
            text = result.chunk.text[: min(len(result.chunk.text), budget)]
            context.append(
                {
                    "file": result.chunk.file_path,
                    "modality": result.chunk.modality,
                    "score": round(result.score, 4),
                    "text": text,
                }
            )
            budget -= len(text)
        return context

    def fallback_answer(self, question: Question, retrieved: list[RetrievedChunk]) -> AnswerResult:
        evidences = top_evidence_files(retrieved, self.settings.max_files_for_question)
        answer = "Not enough data to answer."
        return AnswerResult(
            qid=question.qid,
            answer=answer,
            evidences=evidences,
            diagnostics={"method": "fallback_no_llm"},
        )

    def answer_image_group_with_vision(
        self, question: Question, image_paths: list[Path], file_paths: list[str]
    ) -> AnswerResult | None:
        if not (self.settings.use_vision_llm and self.llm.available and image_paths):
            return None
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Answer the question by inspecting these images. "
                    "Return JSON with answer and short per-image notes. Question: "
                    + question.question
                ),
            }
        ]
        for path in image_paths[:20]:
            content.append({"type": "text", "text": f"Image file: {path.name}"})
            content.append(image_part_from_path(path, self.settings.max_image_side))
        try:
            data = self.llm.json_chat([{"role": "user", "content": content}], max_tokens=1200)
            return AnswerResult(
                qid=question.qid,
                answer=normalize_answer(str(data.get("answer", "")), question.answer_type == "exact_match"),
                evidences=dedupe_keep_order(file_paths),
                diagnostics={"method": "vision_group"},
            )
        except Exception:
            return None


def normalize_answer(answer: str, exact: bool) -> str:
    answer = normalize_text(answer)
    if not answer:
        return ""
    if exact:
        answer = answer.strip().strip('"').strip("'")
        if re.fullmatch(r"-?\d+\.0", answer):
            answer = answer[:-2]
    return answer


def parse_evidence_literal(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []
