from __future__ import annotations

import csv
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from mmdlqa_core.config import Settings
from mmdlqa_core.schema import AnswerResult, Question, RetrievedChunk
from mmdlqa_retrieval.hybrid import top_evidence_files


class DeterministicToolbox:
    def __init__(self, settings: Settings):
        self.settings = settings

    def try_answer(self, question: Question, retrieved: list[RetrievedChunk]) -> AnswerResult | None:
        significant = self.try_count_significant_genes(question, retrieved)
        if significant is not None:
            answer, evidence = significant
            return AnswerResult(
                qid=question.qid,
                answer=str(answer),
                evidences=[evidence],
                diagnostics={"method": "deterministic_xlsx_sheet_count"},
            )

        corr = re.search(
            r"correlation coefficient between the [\"']([^\"']+)[\"'] and [\"']([^\"']+)[\"'] columns",
            question.question,
            flags=re.I,
        )
        if corr:
            answer = self.try_table_correlation(corr.group(1), corr.group(2), retrieved)
            if answer is not None:
                value, evidence = answer
                return AnswerResult(
                    qid=question.qid,
                    answer=value,
                    evidences=[evidence],
                    diagnostics={"method": "deterministic_correlation"},
                )

        q = question.question.casefold()
        if "number_image" in q and "blue digit" in q:
            answer = self.try_count_blue_images(retrieved)
            if answer is not None:
                return AnswerResult(
                    qid=question.qid,
                    answer=str(answer),
                    evidences=top_evidence_files(retrieved, self.settings.max_files_for_question),
                    diagnostics={"method": "color_heuristic"},
                )

        if "number_image" in q and "exactly one digit" in q:
            answer = self.try_count_one_digit_images(retrieved)
            if answer is not None:
                return AnswerResult(
                    qid=question.qid,
                    answer=str(answer),
                    evidences=top_evidence_files(retrieved, self.settings.max_files_for_question),
                    diagnostics={"method": "ocr_digit_heuristic"},
                )

        class_avg = self.try_class_grade_multiple_choice(question, retrieved)
        if class_avg is not None:
            answer, evidence = class_avg
            return AnswerResult(
                qid=question.qid,
                answer=answer,
                evidences=[evidence],
                diagnostics={"method": "deterministic_sql_average_choice"},
            )

        return None

    def try_table_correlation(
        self, col_a: str, col_b: str, retrieved: list[RetrievedChunk]
    ) -> tuple[str, str] | None:
        for result in retrieved:
            path = Path(result.chunk.metadata.get("abs_path") or result.chunk.metadata.get("path") or "")
            if not path.exists():
                path = self.settings.raw_dir / result.chunk.file_path
            if not path.exists() or path.suffix.casefold() not in {".csv", ".xlsx"}:
                continue
            try:
                value = pearson_from_table(path, col_a, col_b)
                if value is not None:
                    return f"{value:.2f}", result.chunk.file_path
            except Exception:
                continue
        return None

    def try_count_significant_genes(
        self, question: Question, retrieved: list[RetrievedChunk]
    ) -> tuple[int, str] | None:
        q = question.question.casefold()
        if "significant genes" not in q:
            return None
        target_sheet_hint = None
        for hint in ["acetyl", "phospho", "proteomics"]:
            if hint in q:
                target_sheet_hint = hint
                break
        for result in retrieved:
            path = Path(str(result.chunk.metadata.get("abs_path", "")))
            if not path.exists() or path.suffix.casefold() != ".xlsx":
                continue
            count = count_nonempty_cells_in_matching_xlsx_sheet(path, target_sheet_hint)
            if count is not None:
                return count, result.chunk.file_path
        return None

    def try_count_blue_images(self, retrieved: list[RetrievedChunk]) -> int | None:
        count = 0
        seen = set()
        found = False
        for result in retrieved:
            if result.chunk.modality != "image" or result.chunk.file_path in seen:
                continue
            seen.add(result.chunk.file_path)
            ratio = result.chunk.metadata.get("blue_pixel_ratio")
            if isinstance(ratio, (int, float)):
                found = True
                if ratio >= 0.015:
                    count += 1
        return count if found else None

    def try_count_one_digit_images(self, retrieved: list[RetrievedChunk]) -> int | None:
        count = 0
        seen = set()
        found = False
        for result in retrieved:
            if result.chunk.modality != "image" or result.chunk.file_path in seen:
                continue
            seen.add(result.chunk.file_path)
            match = re.search(r"OCR text:\s*(.*?)(?:\n[A-Z][A-Za-z ]+:|$)", result.chunk.text, flags=re.S)
            if not match:
                continue
            ocr_text = match.group(1)
            digits = re.findall(r"\d", ocr_text)
            if ocr_text.strip():
                found = True
                if len(digits) == 1:
                    count += 1
        return count if found else None

    def try_class_grade_multiple_choice(
        self, question: Question, retrieved: list[RetrievedChunk]
    ) -> tuple[str, str] | None:
        q = question.question.casefold()
        if "điểm trung bình" not in q and "average" not in q:
            return None
        if "toán" not in q and "math" not in q:
            return None
        class_match = re.search(r"\b\d{2}[a-z]\d\b", question.question, flags=re.I)
        class_name = class_match.group(0).upper() if class_match else ""
        options = parse_multiple_choice_options(question.question)
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        for result in retrieved:
            if result.chunk.file_path.casefold().endswith(".sql") and result.chunk.file_path not in seen:
                candidates.append((result.chunk.file_path, result.chunk.text))
                seen.add(result.chunk.file_path)
        for path in self.settings.raw_dir.rglob("*.sql"):
            rel = path.relative_to(self.settings.raw_dir).as_posix()
            if rel not in seen:
                candidates.append((rel, path.read_text(encoding="utf-8", errors="ignore")))
                seen.add(rel)

        for file_path, sql_text in candidates:
            avg = average_math_score_from_sql(sql_text, class_name)
            if avg is None:
                continue
            if options:
                return min(options.items(), key=lambda kv: abs(kv[1] - avg))[0], file_path
            return f"{avg:.2f}", file_path
        return None


def pearson_from_table(path: Path, col_a: str, col_b: str) -> float | None:
    if path.suffix.casefold() == ".csv":
        rows = read_csv_dicts(path)
    elif path.suffix.casefold() == ".xlsx":
        rows = read_first_xlsx_sheet_dicts(path)
    else:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        try:
            x = float(str(row.get(col_a, "")).strip())
            y = float(str(row.get(col_b, "")).strip())
        except ValueError:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_first_xlsx_sheet_dicts(path: Path) -> list[dict[str, str]]:
    sheets = read_xlsx_sheets(path)
    if not sheets:
        return []
    rows = next(iter(sheets.values()))
    if not rows:
        return []
    headers = rows[0]
    return [{headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))} for row in rows[1:]]


def count_nonempty_cells_in_matching_xlsx_sheet(path: Path, hint: str | None) -> int | None:
    sheets = read_xlsx_sheets(path)
    if not sheets:
        return None
    target_name = None
    if hint:
        for name in sheets:
            if hint in name.casefold():
                target_name = name
                break
    if target_name is None:
        target_name = next(iter(sheets))
    values = [cell for row in sheets[target_name] for cell in row if str(cell).strip()]
    return len(values) if values else None


def read_xlsx_sheets(path: Path) -> dict[str, list[list[str]]]:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    def col_to_idx(ref: str) -> int:
        letters = "".join(ch for ch in ref if ch.isalpha())
        n = 0
        for ch in letters:
            n = n * 26 + ord(ch.upper()) - 64
        return max(0, n - 1)

    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//main:t", ns)))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        out: dict[str, list[list[str]]] = {}
        for sheet in wb.findall("main:sheets/main:sheet", ns):
            name = sheet.attrib["name"]
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = relmap[rid]
            if not target.startswith("xl/"):
                target = "xl/" + target
            root = ET.fromstring(z.read(target))
            rows: list[list[str]] = []
            for row in root.findall("main:sheetData/main:row", ns):
                cells: dict[int, str] = {}
                for c in row.findall("main:c", ns):
                    ref = c.attrib.get("r", "")
                    typ = c.attrib.get("t")
                    value = ""
                    v = c.find("main:v", ns)
                    if v is not None and v.text is not None:
                        value = shared[int(v.text)] if typ == "s" else v.text
                    elif typ == "inlineStr":
                        value = "".join(t.text or "" for t in c.findall(".//main:t", ns))
                    cells[col_to_idx(ref)] = value
                if cells:
                    max_col = max(cells)
                    rows.append([cells.get(i, "") for i in range(max_col + 1)])
            out[name] = rows
        return out


def average_math_score_from_sql(sql_text: str, class_name: str) -> float | None:
    pattern = re.compile(
        r"\(\s*\d+\s*,\s*'[^']+'\s*,\s*'(?P<class>[^']+)'\s*,\s*(?P<math>-?\d+(?:\.\d+)?)\s*,",
        flags=re.I,
    )
    values = []
    for match in pattern.finditer(sql_text):
        if class_name and match.group("class").upper() != class_name:
            continue
        values.append(float(match.group("math")))
    if not values:
        return None
    return sum(values) / len(values)


def parse_multiple_choice_options(question: str) -> dict[str, float]:
    options: dict[str, float] = {}
    for letter, value in re.findall(r"\b([A-D])\.\s*(-?\d+(?:\.\d+)?)", question, flags=re.I):
        options[letter.upper()] = float(value)
    return options
