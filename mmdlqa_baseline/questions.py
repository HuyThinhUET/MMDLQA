from __future__ import annotations

import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .schema import Question
from .utils import json_loads_maybe, normalize_text

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _col_to_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch.upper()) - 64
    return max(0, n - 1)


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", NS):
                shared.append("".join(t.text or "" for t in si.findall(".//main:t", NS)))

        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        first_sheet = workbook.find("main:sheets/main:sheet", NS)
        if first_sheet is None:
            return []
        rid = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = relmap[rid]
        if not target.startswith("xl/"):
            target = "xl/" + target
        sheet = ET.fromstring(z.read(target))

        rows: list[list[str]] = []
        for row in sheet.findall("main:sheetData/main:row", NS):
            cells: dict[int, str] = {}
            for c in row.findall("main:c", NS):
                ref = c.attrib.get("r", "")
                typ = c.attrib.get("t")
                value = ""
                v = c.find("main:v", NS)
                if v is not None and v.text is not None:
                    value = shared[int(v.text)] if typ == "s" else v.text
                elif typ == "inlineStr":
                    value = "".join(t.text or "" for t in c.findall(".//main:t", NS))
                cells[_col_to_idx(ref)] = value
            if cells:
                max_col = max(cells)
                rows.append([cells.get(i, "") for i in range(max_col + 1)])
        return rows


def _clean_id(value: str) -> str:
    value = str(value or "").strip()
    if re.fullmatch(r"\d+\.0", value):
        return value[:-2]
    return value


def _parse_sources(value: str) -> list[str]:
    loaded = json_loads_maybe(value)
    if isinstance(loaded, list):
        return [str(v) for v in loaded]
    if not value.strip():
        return []
    return [v.strip() for v in re.split(r"[,;\n]", value) if v.strip()]


def _rows_to_questions(rows: list[list[str]]) -> list[Question]:
    if not rows:
        return []
    header = [normalize_text(c).casefold() for c in rows[0]]
    aliases = {
        "id": {"id", "stt", "qid", "question_id"},
        "question": {"question", "query", "câu hỏi", "cau hoi"},
        "groundtruth": {"groundtruth", "ground truth", "answer", "gold"},
        "sources": {"data sources", "evidences", "evidence", "sources"},
        "answer_type": {"answer type", "answer_type", "type"},
    }

    def idx(name: str, default: int | None = None) -> int | None:
        for i, h in enumerate(header):
            if h in aliases[name]:
                return i
        return default

    id_i = idx("id", 0)
    q_i = idx("question", 1)
    gt_i = idx("groundtruth")
    src_i = idx("sources")
    type_i = idx("answer_type")
    questions: list[Question] = []
    for row in rows[1:]:
        if q_i is None or q_i >= len(row):
            continue
        question = normalize_text(row[q_i])
        if not question:
            continue
        qid = _clean_id(row[id_i]) if id_i is not None and id_i < len(row) else str(len(questions) + 1)
        questions.append(
            Question(
                qid=qid,
                question=question,
                answer_type=normalize_text(row[type_i]) if type_i is not None and type_i < len(row) else "",
                groundtruth=normalize_text(row[gt_i]) if gt_i is not None and gt_i < len(row) else "",
                data_sources=_parse_sources(row[src_i]) if src_i is not None and src_i < len(row) else [],
            )
        )
    return questions


def load_questions(path: Path) -> list[Question]:
    suffix = path.suffix.casefold()
    if suffix == ".xlsx":
        return _rows_to_questions(_read_xlsx_rows(path))
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [reader.fieldnames or []]
            rows.extend([[row.get(h, "") for h in rows[0]] for row in reader])
        return _rows_to_questions(rows)
    raise ValueError(f"Unsupported questions file: {path}")
