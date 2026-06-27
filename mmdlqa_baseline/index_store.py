from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .extractors import extract_file
from .openrouter import OpenRouterClient
from .schema import Chunk, FileRecord
from .utils import read_jsonl, write_jsonl


def index_paths(settings: Settings) -> tuple[Path, Path]:
    return settings.cache_dir / "files.jsonl", settings.cache_dir / "chunks.jsonl"


def build_index(settings: Settings, *, force: bool = False) -> tuple[list[FileRecord], list[Chunk]]:
    settings.ensure_dirs()
    files_path, chunks_path = index_paths(settings)
    if not force and files_path.exists() and chunks_path.exists():
        return load_index(settings)

    llm = OpenRouterClient(settings)
    records: list[FileRecord] = []
    chunks: list[Chunk] = []
    for path in iter_raw_files(settings.raw_dir):
        record = extract_file(path, settings.raw_dir, settings, llm)
        records.append(record)
        chunks.extend(record.chunks)

    write_jsonl(files_path, [record_to_json(r) for r in records])
    write_jsonl(chunks_path, [asdict(c) for c in chunks])
    return records, chunks


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
