"""Read-only: find which lighting-related categories/questions exist in the data.

Run this FIRST. build_lighting_dataset.py assumes the category names
'lighting type' and 'lighting condition'; this script shows what the local
ShotQA / ShotBench files actually contain so you can confirm or override them.
Nothing is written or modified.

Usage: python scripts/lighting/discover_lighting.py [--keyword light]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.paths import CSV_DIR, SHOTQA_META_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyword", default="light")
    key = parser.parse_args().keyword.lower()

    # ShotQA SFT labels (needs build_labels.py to have been run)
    labels_path = CSV_DIR / "shotqa_labels.csv"
    if labels_path.exists():
        lab = pd.read_csv(labels_path)
        hit = lab[lab["question"].str.lower().str.contains(key, regex=False)]
        print(f"== SFT questions containing '{key}': {len(hit)} rows, {hit['image'].nunique()} images")
        print(hit["question"].value_counts().to_string())
        for question, group in hit.groupby("question"):
            print(f"\n  labels for: {question}")
            print(group["label"].value_counts().to_string())
    else:
        print(f"(missing {labels_path}; run scripts/build_labels.py first)")

    # ShotQA GRPO
    grpo_path = SHOTQA_META_DIR / "grpo.json"
    if grpo_path.exists():
        grpo = json.load(open(grpo_path))
        cats = Counter(str(x.get("_category", "")) for x in grpo)
        print("\n== GRPO categories (all):")
        for cat, n in cats.most_common():
            print(f"  {cat!r}: {n}")
        for cat in [c for c in cats if key in c.lower()]:
            counts = Counter(
                label
                for x in grpo
                if str(x.get("_category", "")) == cat
                for label in x.get("_labels", [])
            )
            print(f"\n  GRPO labels for {cat!r}:")
            for label, n in counts.most_common():
                print(f"    {label}: {n}")
    else:
        print(f"(missing {grpo_path})")

    # ShotBench test annotations (needs make_annotations.py)
    ann_path = CSV_DIR / "annotations.csv"
    if ann_path.exists():
        ann = pd.read_csv(ann_path)
        print("\n== ShotBench categories (all):")
        print(ann["category"].value_counts().to_string())
        sub = ann[ann["category"].str.lower().str.contains(key, regex=False)].copy()
        for cat, group in sub.groupby("category"):
            labels = [json.loads(o)[a] for o, a in zip(group["options"], group["answer"])]
            print(f"\n  ShotBench labels for {cat!r} (n={len(group)}):")
            print(pd.Series(labels).value_counts().to_string())
    else:
        print(f"(missing {ann_path}; run scripts/make_annotations.py first)")

    print(
        "\nIf the category names differ from 'lighting type' / 'lighting condition', pass "
        "them to build_lighting_dataset.py with --dim NAME=category."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
