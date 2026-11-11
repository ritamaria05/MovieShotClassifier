"""Shared repository paths for data preparation scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"
TXT_DIR = ROOT / "txt"
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = DATA_PROCESSED_DIR / "splits"


def _input_dir(name: str) -> Path:
    for candidate in (ROOT / name, DATA_RAW_DIR / name):
        if candidate.exists():
            return candidate
    return DATA_RAW_DIR / name


SHOTQA_META_DIR = _input_dir("ShotQA_meta")
SHOTBENCH_DIR = _input_dir("ShotBench")
SHOTQA_IMAGES_DIR = next(
    (
        candidate
        for candidate in (
            ROOT / "data_shotqa",
            DATA_RAW_DIR / "data_shotqa",
        )
        if candidate.exists()
    ),
    DATA_RAW_DIR / "data_shotqa",
)


def ensure_output_dirs() -> None:
    for directory in (CSV_DIR, TXT_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, SPLITS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
