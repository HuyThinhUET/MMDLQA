from __future__ import annotations

import json
import mimetypes
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from mmdlqa_core.config import Settings
from mmdlqa_core.schema import FileRecord
from mmdlqa_core.utils import normalize_text

from .extractors import make_chunks, make_light_summary, modality_for


def load_cleaned_text_records(settings: Settings) -> tuple[list[FileRecord], set[str]]:
    root = settings.text_cleaning_output_dir
    by_file = root / "by_file"
    if not root.exists() or not by_file.exists():
        return [], set()

    raw_files = list_raw_files(settings.raw_dir)
    records: list[FileRecord] = []
    coverage_keys: set[str] = set()
    for metadata_path in sorted(by_file.glob("*/metadata.json")):
        record = record_from_cleaned_dir(metadata_path.parent, metadata_path, settings, raw_files)
        if record is None:
            continue
        records.append(record)
        coverage_keys.update(coverage_keys_for_record(record))
    return records, coverage_keys


def record_from_cleaned_dir(
    item_dir: Path,
    metadata_path: Path,
    settings: Settings,
    raw_files: list[Path],
) -> FileRecord | None:
    clean_path = item_dir / "clean.txt"
    raw_text_path = item_dir / "raw.txt"
    if not clean_path.exists():
        return None

    metadata = load_json(metadata_path)
    rel_path = normalize_text(str(metadata.get("relative_path") or infer_relative_path(item_dir)))
    if not rel_path:
        rel_path = item_dir.name
    clean_text = normalize_text(clean_path.read_text(encoding="utf-8", errors="ignore"))
    raw_text = normalize_text(raw_text_path.read_text(encoding="utf-8", errors="ignore")) if raw_text_path.exists() else ""
    source_path = find_raw_source_path(rel_path, metadata, settings.raw_dir, raw_files)
    abs_path = str(source_path.resolve()) if source_path else str(clean_path.resolve())
    modality = modality_for(Path(rel_path))

    local_metadata: dict[str, Any] = {
        **metadata,
        "preprocessed": True,
        "source_id": item_dir.name,
        "clean_abs_path": str(clean_path.resolve()),
        "raw_text_abs_path": str(raw_text_path.resolve()) if raw_text_path.exists() else "",
        "raw_source_abs_path": str(source_path.resolve()) if source_path else "",
        "raw_source_found": bool(source_path),
        "raw_text_chars": len(raw_text),
        "clean_text_chars": len(clean_text),
    }
    add_optional_json_metadata(item_dir, local_metadata)

    record = FileRecord(
        file_path=rel_path,
        abs_path=abs_path,
        modality=modality,
        mime_hint=mimetypes.guess_type(rel_path)[0] or "",
        text=clean_text or raw_text or rel_path,
        summary=make_light_summary(rel_path, modality, clean_text or raw_text, local_metadata),
        metadata=local_metadata,
    )
    record.chunks = make_chunks(record, settings)
    return record


def add_optional_json_metadata(item_dir: Path, metadata: dict[str, Any]) -> None:
    ocr_lines = item_dir / "ocr_lines.json"
    ocr_boxes = item_dir / "ocr_boxes.json"
    if ocr_lines.exists():
        lines = load_json(ocr_lines, default=[])
        metadata["ocr_line_count"] = len(lines) if isinstance(lines, list) else 0
        metadata["ocr_lines_abs_path"] = str(ocr_lines.resolve())
    if ocr_boxes.exists():
        boxes = load_json(ocr_boxes, default=[])
        metadata["ocr_box_count"] = len(boxes) if isinstance(boxes, list) else 0
        metadata["ocr_boxes_abs_path"] = str(ocr_boxes.resolve())


def load_json(path: Path, default: Any | None = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {} if default is None else default


def infer_relative_path(item_dir: Path) -> str:
    name = item_dir.name
    return name.rsplit("__", 1)[0] if "__" in name else name


def list_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.rglob("*") if path.is_file())


def find_raw_source_path(
    rel_path: str,
    metadata: dict[str, Any],
    raw_dir: Path,
    raw_files: list[Path],
) -> Path | None:
    direct = raw_dir / rel_path
    if direct.exists():
        return direct

    source_name = Path(str(metadata.get("source_path", ""))).name
    candidates = [rel_path, Path(rel_path).name, source_name]
    for candidate in candidates:
        if not candidate:
            continue
        exact = [path for path in raw_files if path.name == candidate]
        if exact:
            return exact[0]

    rel_key = file_key(rel_path)
    source_key = file_key(source_name)
    keys = [key for key in [rel_key, source_key] if key]
    for path in raw_files:
        raw_key = file_key(path.relative_to(raw_dir).as_posix())
        raw_name_key = file_key(path.name)
        if any(is_probable_same_file(key, raw_key) or is_probable_same_file(key, raw_name_key) for key in keys):
            return path
    fuzzy = find_similar_raw_source(rel_path, raw_dir, raw_files)
    if fuzzy:
        return fuzzy
    return None


def find_similar_raw_source(rel_path: str, raw_dir: Path, raw_files: list[Path]) -> Path | None:
    rel = Path(rel_path)
    rel_parent = rel.parent.as_posix()
    rel_suffix = rel.suffix.casefold()
    rel_stem_key = file_key(rel.stem)
    if len(rel_stem_key) < 8:
        return None

    best_path: Path | None = None
    best_score = 0.0
    for path in raw_files:
        raw_rel = path.relative_to(raw_dir)
        if raw_rel.parent.as_posix() != rel_parent:
            continue
        if path.suffix.casefold() != rel_suffix:
            continue
        raw_stem_key = file_key(path.stem)
        if len(raw_stem_key) < 8:
            continue
        score = SequenceMatcher(None, rel_stem_key, raw_stem_key).ratio()
        if score > best_score:
            best_score = score
            best_path = path
    return best_path if best_score >= 0.68 else None


def coverage_keys_for_record(record: FileRecord) -> set[str]:
    keys: set[str] = set()
    add_coverage_key(keys, "rel", record.file_path)
    add_coverage_key(keys, "name", Path(record.file_path).name)

    source_path = str(record.metadata.get("source_path", ""))
    if source_path:
        add_coverage_key(keys, "name", Path(source_path.replace("\\", "/")).name)

    raw_source = str(record.metadata.get("raw_source_abs_path", ""))
    if raw_source:
        add_coverage_key(keys, "name", Path(raw_source).name)
    return keys


def add_coverage_key(keys: set[str], namespace: str, value: str) -> None:
    key = file_key(value)
    if key:
        keys.add(f"{namespace}:{key}")


def raw_file_covered(path: Path, raw_dir: Path, coverage_keys: set[str]) -> bool:
    rel = path.relative_to(raw_dir).as_posix()
    rel_key = file_key(rel)
    name_key = file_key(path.name)
    return f"rel:{rel_key}" in coverage_keys or f"name:{name_key}" in coverage_keys


def file_key(value: str) -> str:
    value = value.replace("\\", "/").casefold()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"\.[a-z0-9]+$", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def is_probable_same_file(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted([left, right], key=len)
    return len(shorter) >= 8 and longer.startswith(shorter)
