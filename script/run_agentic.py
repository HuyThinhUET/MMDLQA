from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmdlqa_core.config import Settings
from mmdlqa_orchestration.pipeline import run_agentic_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agentic QA workflow and write submission.csv.")
    parser.add_argument("--questions", default=None, help="Questions .xlsx/.csv path.")
    parser.add_argument("--raw-dir", default=None, help="Path to raw data lake directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--submission", default=None, help="Submission CSV path.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild index before answering.")
    parser.add_argument("--limit", type=int, default=None, help="Only answer the first N questions.")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.questions:
        settings.questions_path = Path(args.questions)
    if args.raw_dir:
        settings.raw_dir = Path(args.raw_dir)
    if args.output_dir:
        settings.output_dir = Path(args.output_dir)
        settings.cache_dir = settings.output_dir / "cache"
        settings.submission_path = settings.output_dir / "submission.csv"
    if args.submission:
        settings.submission_path = Path(args.submission)

    run_agentic_pipeline(settings, rebuild_index=args.rebuild_index, limit=args.limit)
    print(f"Wrote {settings.submission_path}")


if __name__ == "__main__":
    main()
