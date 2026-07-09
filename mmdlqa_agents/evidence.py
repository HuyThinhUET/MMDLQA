from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mmdlqa_core.contracts import ClaimEvidence, EvidenceItem
from mmdlqa_core.schema import AnswerResult, RetrievedChunk
from mmdlqa_core.utils import dedupe_keep_order, normalize_text, stable_id


INSUFFICIENT_ANSWER = "Not enough data to answer."


def ledger_from_result(
    source: str,
    result: AnswerResult,
    retrieved: list[RetrievedChunk],
    *,
    supported: bool = True,
) -> list[ClaimEvidence]:
    return fallback_ledger(source, result.answer, result.evidences, retrieved, supported=supported)


def ledger_from_llm_output(
    source: str,
    data: dict[str, Any],
    answer: str,
    evidences: list[str],
    retrieved: list[RetrievedChunk],
) -> list[ClaimEvidence]:
    raw_claims = data.get("claims", data.get("claim_evidence", []))
    if not isinstance(raw_claims, list) or not raw_claims:
        return fallback_ledger(source, answer, evidences, retrieved, supported=bool(evidences))

    ledger: list[ClaimEvidence] = []
    for idx, row in enumerate(raw_claims):
        if not isinstance(row, dict):
            continue
        claim = normalize_text(str(row.get("claim", row.get("statement", ""))))
        if not claim:
            continue
        evidence_files = normalize_file_list(
            row.get("evidence_files", row.get("evidences", row.get("files", [])))
        )
        if not evidence_files:
            evidence_files = evidences
        quotes = normalize_quote_list(row.get("quotes", row.get("quote", [])))
        evidence_items = evidence_items_for_files(evidence_files, retrieved, quotes)
        supported = bool(evidence_items) and all(item.metadata.get("file_in_context") for item in evidence_items)
        ledger.append(
            ClaimEvidence(
                claim_id=stable_id(f"{source}:{idx}:{claim}"),
                claim=claim,
                evidence_items=evidence_items,
                supported=supported,
                source=source,
                notes=normalize_text(str(row.get("notes", ""))),
            )
        )
    return ledger or fallback_ledger(source, answer, evidences, retrieved, supported=bool(evidences))


def fallback_ledger(
    source: str,
    answer: str,
    evidences: list[str],
    retrieved: list[RetrievedChunk],
    *,
    supported: bool,
) -> list[ClaimEvidence]:
    answer = normalize_text(answer)
    if not answer or answer == INSUFFICIENT_ANSWER:
        return []
    evidence_files = dedupe_keep_order(str(item) for item in evidences)
    evidence_items = evidence_items_for_files(evidence_files, retrieved, [])
    return [
        ClaimEvidence(
            claim_id=stable_id(f"{source}:{answer}:{','.join(evidence_files)}"),
            claim=answer,
            evidence_items=evidence_items,
            supported=supported and bool(evidence_items),
            source=source,
            notes="fallback_ledger_from_answer_and_evidence_files",
        )
    ]


def evidence_items_for_files(
    evidence_files: list[str],
    retrieved: list[RetrievedChunk],
    quotes: list[str],
) -> list[EvidenceItem]:
    out: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for index, file_path in enumerate(evidence_files):
        quote = quotes[index] if index < len(quotes) else ""
        item = evidence_item_for_file(file_path, retrieved, quote)
        key = (item.file_path, item.chunk_id)
        if item.file_path and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def evidence_item_for_file(file_path: str, retrieved: list[RetrievedChunk], quote: str = "") -> EvidenceItem:
    file_path = normalize_text(str(file_path))
    quote = normalize_text(str(quote))
    matches = [result for result in retrieved if result.chunk.file_path == file_path]
    chosen = choose_matching_chunk(matches, quote)
    if chosen is None:
        return EvidenceItem(
            file_path=file_path,
            quote=quote,
            metadata={"file_in_context": False, "quote_match": False},
        )
    quote_match = bool(quote and quote.casefold() in chosen.chunk.text.casefold())
    return EvidenceItem(
        file_path=file_path,
        chunk_id=chosen.chunk.chunk_id,
        quote=quote or chosen.chunk.text[:300],
        modality=chosen.chunk.modality,
        score=round(chosen.score, 4),
        metadata={
            "file_in_context": True,
            "quote_match": quote_match,
            "quote_supplied": bool(quote),
            "reasons": chosen.reasons,
        },
    )


def choose_matching_chunk(matches: list[RetrievedChunk], quote: str) -> RetrievedChunk | None:
    if not matches:
        return None
    if quote:
        quote_key = quote.casefold()
        for result in matches:
            if quote_key in result.chunk.text.casefold():
                return result
    return max(matches, key=lambda result: result.score)


def normalize_file_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return dedupe_keep_order(str(item) for item in value if normalize_text(str(item)))


def normalize_quote_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [normalize_text(str(item)) for item in value if normalize_text(str(item))]


def supported_claim_count(ledger: list[ClaimEvidence]) -> int:
    return sum(1 for claim in ledger if claim.supported)


def unsupported_claims(ledger: list[ClaimEvidence]) -> list[ClaimEvidence]:
    return [claim for claim in ledger if not claim.supported]


def ledger_to_dicts(ledger: list[ClaimEvidence]) -> list[dict[str, Any]]:
    return [asdict(item) for item in ledger]
