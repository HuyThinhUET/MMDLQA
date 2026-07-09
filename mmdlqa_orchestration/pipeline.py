from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mmdlqa_agents.answering import Answerer
from mmdlqa_agents.workflow import AgenticAnswerer
from mmdlqa_core.config import Settings
from mmdlqa_core.metrics import QuestionRunTracker, aggregate_question_metrics
from mmdlqa_core.questions import load_questions
from mmdlqa_core.utils import write_jsonl, write_submission_csv
from mmdlqa_preprocess.index_store import build_index, index_metadata, load_index
from mmdlqa_retrieval.hybrid import HybridRetriever
from mmdlqa_retrieval.rag import SentenceRAG


def run_pipeline(settings: Settings, *, rebuild_index: bool = False, limit: int | None = None) -> None:
    settings.ensure_dirs()
    records, chunks = build_index(settings, force=rebuild_index)
    questions = load_questions(settings.questions_path)
    if limit is not None:
        questions = questions[:limit]

    retriever = HybridRetriever(chunks, settings)
    answerer = Answerer(settings)
    output_rows = []
    diagnostics = []

    for question in questions:
        with QuestionRunTracker(settings, question.qid) as tracker:
            with tracker.stage("retrieval"):
                retrieved = retriever.search(question, settings.raw_dir)
            with tracker.stage("answering"):
                result = answerer.answer(question, retrieved)
            result.diagnostics["metrics"] = tracker.snapshot()
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
    if limit is not None:
        questions = questions[:limit]

    rag = SentenceRAG(chunks, settings)
    answerer = AgenticAnswerer(settings, rag)
    output_rows = []
    diagnostics = []

    for question in questions:
        with QuestionRunTracker(settings, question.qid) as tracker:
            result = answerer.answer(question)
            result.diagnostics["metrics"] = tracker.snapshot()
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
