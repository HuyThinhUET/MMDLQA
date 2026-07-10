from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from mmdlqa_core.config import Settings
from mmdlqa_core.openrouter import OpenRouterClient
from mmdlqa_core.schema import Chunk, FileRecord
from mmdlqa_core.utils import read_jsonl, stable_id, write_jsonl

from .cleaned_loader import load_cleaned_text_records, raw_file_covered
from .extractors import extract_file


@dataclass(slots=True)
class RawFileSource:
    path: Path
    root: Path
    source_type: str = "raw_dir"
    archive_path: Path | None = None


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
    raw_sources = (
        list(iter_raw_file_sources(settings))
        if settings.include_raw_fallback or settings.use_text_cleaning_output
        else []
    )
    raw_roots = {source.path.resolve(): source.root for source in raw_sources}

    if settings.use_text_cleaning_output:
        cleaned_records, coverage_keys = load_cleaned_text_records(
            settings,
            raw_files=[source.path for source in raw_sources],
            raw_roots=raw_roots,
        )
        records.extend(cleaned_records)
        for record in cleaned_records:
            chunks.extend(record.chunks)

    if not settings.include_raw_fallback:
        write_index(settings, records, chunks)
        return records, chunks

    for source in raw_sources:
        if settings.use_text_cleaning_output and raw_file_covered(source.path, source.root, coverage_keys):
            continue
        record = extract_file(source.path, source.root, settings, llm)
        annotate_raw_source(record, source)
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
        "raw_inputs": raw_input_signature(settings.raw_dir),
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


def iter_raw_file_sources(settings: Settings) -> Iterable[RawFileSource]:
    for path in iter_raw_files(settings.raw_dir):
        if is_zip_archive(path):
            continue
        yield RawFileSource(path=path, root=settings.raw_dir)

    yield from prepare_raw_archive_sources(settings)


def iter_raw_files(raw_dir: Path):
    if not raw_dir.exists():
        return
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            yield path


def prepare_raw_archive_sources(settings: Settings) -> list[RawFileSource]:
    archives = [path for path in iter_raw_files(settings.raw_dir) if is_zip_archive(path)]
    if not archives:
        return []
    archive_root = settings.cache_dir / "raw_unzipped"
    archive_root.mkdir(parents=True, exist_ok=True)
    sources: list[RawFileSource] = []
    seen: set[Path] = set()
    for archive in archives:
        for path in extract_zip_archive(archive, archive_root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            sources.append(
                RawFileSource(
                    path=path,
                    root=archive_root,
                    source_type="zip_archive",
                    archive_path=archive,
                )
            )
    return sources


def extract_zip_archive(archive: Path, target_root: Path) -> list[Path]:
    marker = target_root / ".archive_markers" / f"{archive.stem}-{archive_signature(archive)}.json"
    members = zip_member_targets(archive, target_root)
    if marker.exists():
        existing = [target for _, target in members if target.exists()]
        if len(existing) == len(members):
            return existing
    marker.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info, target in members:
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
    marker.write_text(json.dumps({"archive": str(archive), "signature": archive_signature(archive)}), encoding="utf-8")
    return [target for _, target in members if target.exists()]


def zip_member_targets(archive: Path, target_root: Path) -> list[tuple[zipfile.ZipInfo, Path]]:
    root = target_root.resolve()
    members: list[tuple[zipfile.ZipInfo, Path]] = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir() or should_skip_archive_member(info.filename):
                continue
            target = (target_root / normalized_member_path(info.filename)).resolve()
            if is_within_directory(target, root):
                members.append((info, target))
    return members


def normalized_member_path(name: str) -> Path:
    return Path(name.replace("\\", "/").lstrip("/"))


def should_skip_archive_member(name: str) -> bool:
    normalized = name.replace("\\", "/").lstrip("/")
    parts = [part for part in normalized.split("/") if part]
    return not parts or any(part.startswith(".") for part in parts) or parts[0] == "__MACOSX"


def is_within_directory(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def is_zip_archive(path: Path) -> bool:
    return path.suffix.casefold() == ".zip"


def archive_signature(path: Path) -> str:
    stat = path.stat()
    return stable_id(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}", length=12)


def raw_input_signature(raw_dir: Path) -> list[dict]:
    if not raw_dir.exists():
        return []
    signature = []
    for path in iter_raw_files(raw_dir):
        stat = path.stat()
        signature.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return signature


def annotate_raw_source(record: FileRecord, source: RawFileSource) -> None:
    record.metadata["raw_source_type"] = source.source_type
    if source.archive_path:
        record.metadata["archive_path"] = source.archive_path.name
    for chunk in record.chunks:
        chunk.metadata["raw_source_type"] = source.source_type
        if source.archive_path:
            chunk.metadata["archive_path"] = source.archive_path.name


def record_to_json(record: FileRecord) -> dict:
    data = asdict(record)
    data["chunks"] = [asdict(c) for c in record.chunks]
    return data


def file_from_json(row: dict) -> FileRecord:
    chunks = [Chunk(**c) for c in row.pop("chunks", [])]
    record = FileRecord(**row)
    record.chunks = chunks
    return record
