from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Question:
    qid: str
    question: str
    answer_type: str = ""
    groundtruth: str = ""
    data_sources: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    file_path: str
    modality: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FileRecord:
    file_path: str
    abs_path: str
    modality: str
    mime_hint: str = ""
    text: str = ""
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def path_obj(self) -> Path:
        return Path(self.abs_path)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnswerResult:
    qid: str
    answer: str
    evidences: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)
