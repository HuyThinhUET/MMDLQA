from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mmdlqa_agents.answering import Answerer
from mmdlqa_agents.workflow import AgenticAnswerer
from mmdlqa_core.config import Settings
from mmdlqa_core.metrics import QuestionRunTracker, aggregate_question_metrics
from mmdlqa_core.openrouter import sanitize_api_key
from mmdlqa_core.questions import load_questions
from mmdlqa_core.utils import write_jsonl, write_submission_csv
from mmdlqa_preprocess.index_store import build_index, index_metadata, load_index
from mmdlqa_retrieval.hybrid import HybridRetriever
from mmdlqa_retrieval.rag import SentenceRAG


def run_pipeline(settings: Settings, *, rebuild_index: bool = False, limit: int | None = None) -> None:
    settings.ensure_dirs()
    records, chunks = build_index(settings, force=rebuild_index)
    questions = load_questions(settings.questions_path)
    questions = apply_question_limit(questions, settings, limit)

    retriever = HybridRetriever(chunks, settings)
    answerer = Answerer(settings)
    output_rows = []
    diagnostics = []
    print_run_preflight(settings)

    for question in questions:
        with QuestionRunTracker(settings, question.qid) as tracker:
            with tracker.stage("retrieval"):
                retrieved = retriever.search(question, settings.raw_dir)
            with tracker.stage("answering"):
                result = answerer.answer(question, retrieved)
            result.diagnostics["metrics"] = tracker.snapshot()
        print_question_progress(settings, result)
        output_rows.append({"id": result.qid, "answer": result.answer, "evidences": result.evidences})
        diagnostics.append(
            {
                "question": asdict(question),
                "answer": asdict(result),
                "retrieved": [
                    {
                        "file": r.chunk.file_path,
                        "chunk_id": r.chunk.chunk_id,
                        "score": r.score,
                        "modality": r.chunk.modality,
                        "preview": r.chunk.text[:300],
                    }
                    for r in retrieved
                ],
            }
        )

    write_submission_csv(settings.submission_path, output_rows)
    write_jsonl(settings.output_dir / "diagnostics.jsonl", diagnostics)
    summary = {
        "mode": "baseline",
        "questions": len(questions),
        "indexed_files": len(records),
        "indexed_chunks": len(chunks),
        "index": index_metadata(settings, records, chunks),
        "submission": str(settings.submission_path),
        "metrics": aggregate_question_metrics(diagnostics),
    }
    (settings.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_only(settings: Settings, *, force: bool = False) -> None:
    settings.ensure_dirs()
    records, chunks = build_index(settings, force=force)
    summary = {
        "indexed_files": len(records),
        "indexed_chunks": len(chunks),
        "index": index_metadata(settings, records, chunks),
        "cache_dir": str(settings.cache_dir),
    }
    (settings.output_dir / "index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def require_existing_index(settings: Settings):
    return load_index(settings)


def run_agentic_pipeline(settings: Settings, *, rebuild_index: bool = False, limit: int | None = None) -> None:
    settings.ensure_dirs()
    records, chunks = build_index(settings, force=rebuild_index)
    questions = load_questions(settings.questions_path)
    questions = apply_question_limit(questions, settings, limit)

    rag = SentenceRAG(chunks, settings)
    answerer = AgenticAnswerer(settings, rag)
    output_rows = []
    diagnostics = []
    print_run_preflight(settings)

    for question in questions:
        with QuestionRunTracker(settings, question.qid) as tracker:
            result = answerer.answer(question)
            result.diagnostics["metrics"] = tracker.snapshot()
        print_question_progress(settings, result)
        output_rows.append({"id": result.qid, "answer": result.answer, "evidences": result.evidences})
        diagnostics.append(
            {
                "question": asdict(question),
                "answer": asdict(result),
                "agentic": result.diagnostics.get("agentic", {}),
            }
        )

    write_submission_csv(settings.submission_path, output_rows)
    write_jsonl(settings.output_dir / "diagnostics.jsonl", diagnostics)
    summary = {
        "mode": "agentic",
        "questions": len(questions),
        "indexed_files": len(records),
        "indexed_chunks": len(chunks),
        "index": index_metadata(settings, records, chunks),
        "submission": str(settings.submission_path),
        "metrics": aggregate_question_metrics(diagnostics),
    }
    (settings.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def print_question_progress(settings: Settings, result) -> None:
    if not settings.print_question_metrics:
        return
    metrics = result.diagnostics.get("metrics", {})
    elapsed = float(metrics.get("elapsed_sec", 0.0) or 0.0)
    calls = int(metrics.get("llm_call_count", 0) or 0)
    failed = int(metrics.get("failed_llm_call_count", 0) or 0)
    cost = float(metrics.get("total_estimated_cost_usd", 0.0) or 0.0)
    answer = str(result.answer).replace("\n", " ")[:100]
    print(
        f"[qid={result.qid}] {elapsed:.2f}s | llm_calls={calls} | failed_llm={failed} | "
        f"cost=${cost:.5f} | answer={answer}",
        flush=True,
    )
    if failed and metrics.get("first_llm_error"):
        print(f"[qid={result.qid} llm_error] {str(metrics['first_llm_error'])[:500]}", flush=True)


def print_run_preflight(settings: Settings) -> None:
    if not settings.print_question_metrics:
        return
    key_present = bool(sanitize_api_key(settings.openrouter_api_key))
    llm_available = key_present and settings.use_llm
    print(
        "[run] "
        f"use_llm={settings.use_llm} | openrouter_key_present={key_present} | "
        f"llm_available={llm_available} | evidence_scanner={settings.use_evidence_scanner} | "
        f"max_questions={settings.max_questions}",
        flush=True,
    )
    if not llm_available:
        print(
            "[run warning] LLM is not available. This is a dry-run style execution; "
            "planner/MoE/critic/evidence scanner LLM calls will be skipped.",
            flush=True,
        )


def apply_question_limit(questions, settings: Settings, limit: int | None = None):
    max_questions = settings.max_questions if limit is None else limit
    if max_questions < 0:
        return questions
    return questions[:max_questions]
