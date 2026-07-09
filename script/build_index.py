from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmdlqa_core.config import Settings
from mmdlqa_orchestration.pipeline import build_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/cache the data-lake index.")
    parser.add_argument("--raw-dir", default=None, help="Path to raw data lake directory.")
    parser.add_argument("--text-cleaning-output", default=None, help="Path to preprocessed text_cleaning_output.")
    parser.add_argument("--no-text-cleaning-output", action="store_true", help="Index raw files only.")
    parser.add_argument("--no-raw-fallback", action="store_true", help="Index cleaned output only.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--cache-dir", default=None, help="Cache directory.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists.")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.raw_dir:
        settings.raw_dir = Path(args.raw_dir)
    if args.text_cleaning_output:
        settings.text_cleaning_output_dir = Path(args.text_cleaning_output)
        settings.use_text_cleaning_output = True
    if args.no_text_cleaning_output:
        settings.use_text_cleaning_output = False
    if args.no_raw_fallback:
        settings.include_raw_fallback = False
    if args.output_dir:
        settings.output_dir = Path(args.output_dir)
        settings.cache_dir = settings.output_dir / "cache"
    if args.cache_dir:
        settings.cache_dir = Path(args.cache_dir)

    build_only(settings, force=args.force)
    print(f"Indexed data lake into {settings.cache_dir}")


if __name__ == "__main__":
    main()
