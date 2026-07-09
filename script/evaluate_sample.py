from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmdlqa_core.config import Settings
from mmdlqa_core.questions import load_questions
from mmdlqa_core.utils import normalize_text


def norm(value: str) -> str:
    value = normalize_text(str(value)).strip().strip('"').strip("'")
    if value.endswith(".0") and value[:-2].replace("-", "").isdigit():
        value = value[:-2]
    return value.casefold()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sample exact-match rows locally.")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--submission", default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    questions_path = Path(args.questions) if args.questions else settings.questions_path
    submission_path = Path(args.submission) if args.submission else settings.submission_path
    questions = {q.qid: q for q in load_questions(questions_path)}

    rows = []
    with submission_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    exact_total = 0
    exact_correct = 0
    judged = []
    for row in rows:
        q = questions.get(str(row.get("id", "")))
        if not q:
            continue
        answer = row.get("answer", "")
        evidences_raw = row.get("evidences", "[]")
        try:
            evidences = json.loads(evidences_raw)
        except Exception:
            evidences = []
        if q.answer_type == "exact_match":
            exact_total += 1
            ok = norm(answer) == norm(q.groundtruth)
            exact_correct += int(ok)
            judged.append((q.qid, ok, answer, q.groundtruth, evidences))

    print(f"Exact-match sample score: {exact_correct}/{exact_total}")
    for qid, ok, answer, truth, evidences in judged:
        mark = "OK" if ok else "MISS"
        print(f"{mark}\t{qid}\tpred={answer!r}\tgold={truth!r}\tevidences={evidences}")


if __name__ == "__main__":
    main()
