"""Place the released dataset files under code/data/ with the names the scripts expect.

The released files live in the repository's top-level data/ folder. This script copies them
into code/data/ under the file names used by the pipeline scripts, decompressing the gzipped
model solutions on the way. Nothing is downloaded.

Usage (from the code/ folder):
    python3 scripts/00_fetch_released_data.py

Or point it at another copy of the released files:
    python3 scripts/00_fetch_released_data.py --local path/to/numina-hard-guess-benchmark
"""
from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent.parent / "data"
JUDGES = ["gpt-5.4", "gemini-3.1-pro", "qwen3-235b"]
STAGES = {"stage_a": 1000, "stage_b": 3000}


def file_map(stage: str, n: int) -> dict[str, str]:
    """Released name -> name used by the pipeline scripts."""
    names = {
        f"candidate_problems_{n}.json": f"{stage}_{n}.json",
        "excluded_bad_data_pids.json": f"{stage}_excluded_pids.json",
        "guessability_labels_gemini-3.1-pro.json": f"{stage}_rubric_v2.json",
        "guessability_labels_gpt-5.4_second_judge.json": f"{stage}_rubric_v2_gpt.json",
        "weak_canary_samples.json": f"{stage}_canary.json",
        "geometry_labels.json": f"{stage}_geometry.json",
        "static_rubrics.json": f"{stage}_rubrics.json",
        "model_solutions.json": f"{stage}_inference.json",
        "answer_verification.json": f"{stage}_verification.json",
    }
    names.update({f"grading_{j}.json": f"{stage}_grading_{j}.json" for j in JUDGES})
    return names


def copy_released(src: Path, dst: Path) -> bool:
    """Copy src to dst; fall back to src + '.gz' and decompress it. Return True if copied."""
    if src.exists():
        shutil.copyfile(src, dst)
        return True
    gz = src.with_name(src.name + ".gz")
    if gz.exists():
        with gzip.open(gz, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, default=REPO_ROOT,
                        help="root of a copy of the released files (default: this repository)")
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    copied = 0
    for stage, n in STAGES.items():
        for released, local in file_map(stage, n).items():
            src = args.local / "data" / stage / released
            if copy_released(src, DATA / local):
                copied += 1
            else:
                print(f"missing (skipped): {src}")
    print(f"copied {copied} files into {DATA}")


if __name__ == "__main__":
    main()
