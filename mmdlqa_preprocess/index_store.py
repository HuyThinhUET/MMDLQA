from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mmdlqa_core.config import Settings
from mmdlqa_core.openrouter import OpenRouterClient
from mmdlqa_core.schema import Chunk, FileRecord
from mmdlqa_core.utils import read_jsonl, write_jsonl

from .cleaned_loader import load_cleaned_text_records, raw_file_covered
from .extractors import extract_file


def index_paths(settings: Settings) -> tuple[Path, Path]:
    return settings.cache_dir / "files.jsonl", settings.cache_dir / "chunks.jsonl"


def index_meta_path(settings: Settings) -> Path:
    return settings.cache_dir / "index_meta.json"


def build_index(settings: Settings, *, force: bool = False) -> tuple[list[FileRecord], list[Chunk]]:
    settings.ensure_dirs()
    files_path, chunks_path = index_paths(settings)
    if not force and files_path.exists() and chunks_path.exists() and cache_matches_settings(settings):
        return load_index(settings)

    llm = OpenRouterClient(settings)
    records: list[FileRecord] = []
    chunks: list[Chunk] = []
    coverage_keys: set[str] = set()

    if settings.use_text_cleaning_output:
        cleaned_records, coverage_keys = load_cleaned_text_records(settings)
        records.extend(cleaned_records)
        for record in cleaned_records:
            chunks.extend(record.chunks)

    if not settings.include_raw_fallback:
        write_index(settings, records, chunks)
        return records, chunks

    for path in iter_raw_files(settings.raw_dir):
        if settings.use_text_cleaning_output and raw_file_covered(path, settings.raw_dir, coverage_keys):
            continue
        record = extract_file(path, settings.raw_dir, settings, llm)
        records.append(record)
        chunks.extend(record.chunks)

    write_index(settings, records, chunks)
    return records, chunks


def write_index(settings: Settings, records: list[FileRecord], chunks: list[Chunk]) -> None:
    files_path, chunks_path = index_paths(settings)
    write_jsonl(files_path, [record_to_json(r) for r in records])
    write_jsonl(chunks_path, [asdict(c) for c in chunks])
    meta = index_metadata(settings, records, chunks)
    index_meta_path(settings).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_matches_settings(settings: Settings) -> bool:
    meta_path = index_meta_path(settings)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("settings") == index_settings_signature(settings)


def index_settings_signature(settings: Settings) -> dict:
    return {
        "raw_dir": str(settings.raw_dir),
        "text_cleaning_output_dir": str(settings.text_cleaning_output_dir),
        "use_text_cleaning_output": settings.use_text_cleaning_output,
        "include_raw_fallback": settings.include_raw_fallback,
        "chunk_size_chars": settings.chunk_size_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
    }


def index_metadata(settings: Settings, records: list[FileRecord], chunks: list[Chunk]) -> dict:
    preprocessed = sum(1 for record in records if record.metadata.get("preprocessed"))
    return {
        "settings": index_settings_signature(settings),
        "indexed_files": len(records),
        "indexed_chunks": len(chunks),
        "preprocessed_files": preprocessed,
        "raw_fallback_files": len(records) - preprocessed,
    }


def load_index(settings: Settings) -> tuple[list[FileRecord], list[Chunk]]:
    files_path, chunks_path = index_paths(settings)
    records = [file_from_json(row) for row in read_jsonl(files_path)]
    chunks = [Chunk(**row) for row in read_jsonl(chunks_path)]
    return records, chunks


def iter_raw_files(raw_dir: Path):
    if not raw_dir.exists():
        return
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            yield path


def record_to_json(record: FileRecord) -> dict:
    data = asdict(record)
    data["chunks"] = [asdict(c) for c in record.chunks]
    return data


def file_from_json(row: dict) -> FileRecord:
    chunks = [Chunk(**c) for c in row.pop("chunks", [])]
    record = FileRecord(**row)
    record.chunks = chunks
    return record
