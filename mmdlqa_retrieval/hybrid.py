from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mmdlqa_core.config import Settings
from mmdlqa_core.schema import Chunk, Question, RetrievedChunk
from mmdlqa_core.utils import dedupe_keep_order, normalize_text, tokenize


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], settings: Settings):
        self.chunks = chunks
        self.settings = settings
        self.doc_texts = [document_text(c) for c in chunks]
        self.doc_tokens = [tokenize(text) for text in self.doc_texts]
        self.df = Counter()
        for toks in self.doc_tokens:
            self.df.update(set(toks))
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(1, len(self.doc_tokens))
        self.semantic = SemanticIndex(self.doc_texts, settings) if settings.use_sentence_transformers else None

    def search(self, question: Question, raw_dir: Path, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieve_top_k
        query_tokens = tokenize(question.question)
        query_text = query_with_sources(question)
        mentioned = mentioned_paths(query_text, raw_dir)
        semantic_scores = self.semantic.score(query_text) if self.semantic else {}
        scores: list[RetrievedChunk] = []
        for idx, chunk in enumerate(self.chunks):
            reasons: list[str] = []
            bm25 = self._bm25(query_tokens, idx)
            path_score = path_match_score(chunk.file_path, query_text, mentioned)
            if path_score:
                reasons.append("path/folder mention")
            path_overlap = path_token_overlap_score(chunk.file_path, query_tokens)
            fuzzy_path = fuzzy_path_score(chunk, query_text)
            if fuzzy_path:
                reasons.append("fuzzy path/source match")
            phrase = phrase_match_score(chunk, query_text)
            if phrase:
                reasons.append("phrase match")
            metadata = metadata_hint_score(chunk, query_text)
            modality_score = modality_hint_score(chunk.modality, query_text)
            semantic_score = semantic_scores.get(idx, 0.0)
            if semantic_score:
                reasons.append("semantic match")
            score = bm25 + path_score + path_overlap + fuzzy_path + phrase + metadata + modality_score + semantic_score
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


def query_with_sources(question: Question) -> str:
    parts = [question.question]
    if question.data_sources:
        parts.append(" ".join(question.data_sources))
    return "\n".join(parts)


def document_text(chunk: Chunk) -> str:
    metadata = chunk.metadata or {}
    metadata_parts = [
        chunk.file_path,
        chunk.modality,
        str(metadata.get("source_id", "")),
        str(metadata.get("extract_method", "")),
        str(metadata.get("extension", "")),
        " ".join(map(str, metadata.get("quality_flags", []) or [])),
        str(metadata.get("source_path", "")),
    ]
    return normalize_text("\n".join([*metadata_parts, chunk.text]))


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


def phrase_match_score(chunk: Chunk, question: str) -> float:
    q = normalize_text(question).casefold()
    if not q:
        return 0.0
    text = chunk.text.casefold()
    score = 0.0
    quoted = [item.casefold() for item in re.findall(r"['\"]([^'\"]{4,120})['\"]", question)]
    for phrase in quoted:
        if phrase and phrase in text:
            score += 5.0
    stem = Path(chunk.file_path).stem.casefold()
    if stem and len(stem) >= 4 and stem in q:
        score += 3.0
    return min(score, 12.0)


def fuzzy_path_score(chunk: Chunk, question: str) -> float:
    query_key = compact_key(question)
    if not query_key:
        return 0.0
    candidates = [
        chunk.file_path,
        Path(chunk.file_path).name,
        Path(chunk.file_path).stem,
        str(chunk.metadata.get("source_id", "")),
        str(chunk.metadata.get("source_path", "")),
    ]
    best = 0.0
    for value in candidates:
        key = compact_key(value)
        if len(key) < 4:
            continue
        if key in query_key:
            best = max(best, 7.0)
        else:
            overlap = char_ngram_jaccard(query_key, key)
            if overlap >= 0.18:
                best = max(best, min(5.0, overlap * 12.0))
    return best


def metadata_hint_score(chunk: Chunk, question: str) -> float:
    q = question.casefold()
    metadata = chunk.metadata or {}
    score = 0.0
    method = str(metadata.get("extract_method", "")).casefold()
    if "ocr" in method and any(hint in q for hint in ["ocr", "image", "ảnh", "jpg", "png"]):
        score += 1.0
    if metadata.get("preprocessed"):
        score += 0.25
    if metadata.get("quality_flags"):
        score -= 0.2
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


def compact_key(value: str) -> str:
    value = normalize_text(value).casefold()
    return re.sub(r"[^0-9a-zA-Z\u0080-\uffff]+", "", value)


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    if len(a) < n or len(b) < n:
        return 0.0
    grams_a = {a[i : i + n] for i in range(len(a) - n + 1)}
    grams_b = {b[i : i + n] for i in range(len(b) - n + 1)}
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


def top_evidence_files(results: list[RetrievedChunk], limit: int) -> list[str]:
    by_file: dict[str, float] = defaultdict(float)
    for result in results:
        by_file[result.chunk.file_path] = max(by_file[result.chunk.file_path], result.score)
    ranked = sorted(by_file.items(), key=lambda kv: kv[1], reverse=True)
    return [path for path, _ in ranked[:limit]]


class SemanticIndex:
    def __init__(self, doc_texts: list[str], settings: Settings):
        self.available = False
        self.embeddings: Any = None
        self.model: Any = None
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            model_name = getattr(settings, "sentence_transformer_model", "sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer(model_name)
            self.embeddings = self.model.encode(
                [text[:4000] for text in doc_texts],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.np = np
            self.available = True
        except Exception:
            self.available = False

    def score(self, query: str) -> dict[int, float]:
        if not self.available:
            return {}
        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        similarities = self.embeddings @ query_embedding
        if len(similarities) == 0:
            return {}
        top_count = min(80, len(similarities))
        top_indices = self.np.argpartition(-similarities, range(top_count))[:top_count]
        return {int(idx): max(0.0, float(similarities[idx])) * 8.0 for idx in top_indices}
