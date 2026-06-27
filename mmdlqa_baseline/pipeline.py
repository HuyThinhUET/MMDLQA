from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .answering import Answerer
from .config import Settings
from .index_store import build_index, load_index
from .questions import load_questions
from .retrieval import HybridRetriever
from .utils import write_jsonl, write_submission_csv


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
        retrieved = retriever.search(question, settings.raw_dir)
        result = answerer.answer(question, retrieved)
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
        "questions": len(questions),
        "indexed_files": len(records),
        "indexed_chunks": len(chunks),
        "submission": str(settings.submission_path),
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
        "cache_dir": str(settings.cache_dir),
    }
    (settings.output_dir / "index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def require_existing_index(settings: Settings):
    return load_index(settings)
