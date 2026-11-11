"""Run the data-preparation pipeline through leakage-safe split generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from paths import CSV_DIR, DATA_PROCESSED_DIR, TXT_DIR, ensure_output_dirs


ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    ("Create ShotBench annotations", "make_annotations.py"),
    ("Parse ShotQA labels", "build_labels.py"),
    ("Consolidate dimension labels", "final_labels.py"),
    ("Check metadata", "check_meta.py"),
    ("Check dataset overlap", "check_overlap.py"),
    ("Verify local data", "verify.py"),
    ("Build leakage-safe splits", "build_dataset.py"),
)


def main() -> int:
    ensure_output_dirs()
    print(f"Running data preparation from {ROOT}")
    for number, (name, script) in enumerate(STEPS, start=1):
        command = [sys.executable, str(ROOT / "scripts" / script)]
        print(f"\n[{number}/{len(STEPS)}] {name}: {script}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except FileNotFoundError as error:
            print(f"ERROR: could not start {script}: {error}", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as error:
            print(
                f"ERROR: step '{name}' failed with exit code {error.returncode}. "
                f"Command: {' '.join(command)}",
                file=sys.stderr,
            )
            return error.returncode or 1

    print(
        "\nPreparation complete. The runner stops after train/validation/test "
        f"split generation in {DATA_PROCESSED_DIR / 'splits'}; it does not "
        "extract embeddings or run evaluation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
