from __future__ import annotations

from pathlib import Path

from mmdlqa_core.config import Settings
from mmdlqa_core.contracts import RagQuery
from mmdlqa_core.schema import Chunk, Question, RetrievedChunk

from .hybrid import HybridRetriever


class SentenceRAG:
    """Adapter that keeps the RAG boundary as one natural-language sentence."""

    def __init__(
        self,
        chunks: list[Chunk],
        settings: Settings,
        *,
        retriever: HybridRetriever | None = None,
    ):
        self.settings = settings
        self.retriever = retriever or HybridRetriever(chunks, settings)

    def search_sentence(
        self,
        query: str | RagQuery,
        *,
        raw_dir: Path | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        rag_query = query if isinstance(query, RagQuery) else RagQuery(sentence=query)
        question = Question(
            qid=rag_query.step_id or "rag",
            question=rag_query.sentence,
            answer_type=str(rag_query.metadata.get("answer_type", "")),
        )
        limit = rag_query.top_k if rag_query.top_k is not None else top_k
        return self.retriever.search(question, raw_dir or self.settings.raw_dir, top_k=limit)
