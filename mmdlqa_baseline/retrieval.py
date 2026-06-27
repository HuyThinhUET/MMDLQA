from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import Settings
from .schema import Chunk, Question, RetrievedChunk
from .utils import dedupe_keep_order, tokenize


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], settings: Settings):
        self.chunks = chunks
        self.settings = settings
        self.doc_tokens = [tokenize(c.file_path + "\n" + c.text) for c in chunks]
        self.df = Counter()
        for toks in self.doc_tokens:
            self.df.update(set(toks))
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(1, len(self.doc_tokens))

    def search(self, question: Question, raw_dir: Path, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieve_top_k
        query_tokens = tokenize(question.question)
        mentioned = mentioned_paths(question.question, raw_dir)
        scores: list[RetrievedChunk] = []
        for idx, chunk in enumerate(self.chunks):
            reasons: list[str] = []
            bm25 = self._bm25(query_tokens, idx)
            path_score = path_match_score(chunk.file_path, question.question, mentioned)
            if path_score:
                reasons.append("path/folder mention")
            path_overlap = path_token_overlap_score(chunk.file_path, query_tokens)
            modality_score = modality_hint_score(chunk.modality, question.question)
            score = bm25 + path_score + path_overlap + modality_score
            if score > 0:
                scores.append(RetrievedChunk(chunk=chunk, score=score, reasons=reasons))
        scores.sort(key=lambda r: r.score, reverse=True)
        return scores[:top_k]

    def _bm25(self, query_tokens: list[str], doc_idx: int) -> float:
        if not query_tokens or not self.doc_tokens:
            return 0.0
        toks = self.doc_tokens[doc_idx]
        tf = Counter(toks)
        dl = len(toks)
        k1 = 1.5
        b = 0.75
        score = 0.0
        n_docs = len(self.doc_tokens)
        for term in query_tokens:
            if term not in tf:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / max(1, self.avgdl))
            score += idf * (tf[term] * (k1 + 1) / denom)
        return score


def mentioned_paths(question: str, raw_dir: Path) -> list[str]:
    mentions = []
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
    quoted += re.findall(r"\b[\w./ -]+\.(?:csv|xlsx|xls|pdf|txt|md|html|png|jpg|jpeg|pptx?|sql|m4a|mp3|wav|mp4)\b", question, flags=re.I)
    for item in quoted:
        item = item.strip()
        if item:
            mentions.append(item.replace("\\", "/"))
    folder_like = re.findall(r"\b[\w-]+_[\w-]+\b|\b[\w-]+/[^\s,;:]+", question)
    mentions.extend(m.replace("\\", "/") for m in folder_like)
    existing = []
    for mention in dedupe_keep_order(mentions):
        if "*" in mention:
            existing.append(mention)
            continue
        if (raw_dir / mention).exists():
            existing.append(mention)
    return dedupe_keep_order(existing or mentions)


def path_match_score(file_path: str, question: str, mentions: list[str]) -> float:
    q = question.casefold().replace("\\", "/")
    fp = file_path.casefold()
    score = 0.0
    for mention in mentions:
        m = mention.casefold().strip("*")
        if not m:
            continue
        if m in fp or m in q and Path(fp).name.casefold() in q:
            score += 8.0
    name = Path(fp).name
    stem = Path(fp).stem
    if name in q:
        score += 6.0
    elif stem and stem in q:
        score += 4.0
    return score


def modality_hint_score(modality: str, question: str) -> float:
    q = question.casefold()
    hints = {
        "image": ["image", "images", "ảnh", "jpg", "png", "digit", "blue"],
        "audio": ["audio", "meeting", "m4a", "mp3", "âm thanh"],
        "video": ["video", "mp4"],
        "table": ["csv", "xlsx", "table", "correlation", "average", "sum", "count", "columns"],
        "document": ["pdf", "ppt", "document", "tài liệu", "slide"],
    }
    return 1.5 if any(h in q for h in hints.get(modality, [])) else 0.0


def path_token_overlap_score(file_path: str, query_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    stop = {
        "the",
        "and",
        "or",
        "in",
        "of",
        "for",
        "to",
        "a",
        "an",
        "what",
        "which",
        "how",
        "many",
        "của",
        "là",
        "và",
        "có",
        "cho",
        "tôi",
        "trong",
        "nào",
        "nhiêu",
    }
    q = {t for t in query_tokens if len(t) >= 3 and t not in stop}
    path_tokens = {t for t in tokenize(file_path) if len(t) >= 3}
    return min(8.0, 2.0 * len(q & path_tokens))


def top_evidence_files(results: list[RetrievedChunk], limit: int) -> list[str]:
    by_file: dict[str, float] = defaultdict(float)
    for result in results:
        by_file[result.chunk.file_path] = max(by_file[result.chunk.file_path], result.score)
    ranked = sorted(by_file.items(), key=lambda kv: kv[1], reverse=True)
    return [path for path, _ in ranked[:limit]]
