"""Build film-grouped, leakage-safe train/val/test splits for lighting labels.

Mirrors final_labels.py + build_dataset.py, but for lighting dimensions and with
its own output names, so existing shot_size / framing files are never touched.

Per dimension it:
  1. collects ShotQA SFT + GRPO labels for the category and joins film metadata;
  2. writes csv/labels_<dim>.csv and txt/lista_lighting_extrair.txt (the images
     you must extract from the ShotQA archive: lista_extrair.txt does NOT
     include lighting images);
  3. drops multi-label training rows and training films that appear in ShotBench;
  4. drops rows whose image file is missing on disk (and reports how many);
  5. splits train/val 85/15 grouped by film (random_state=42);
  6. writes data/processed/splits/{train,val,test}_<dim>.csv.

Test labels come from ShotBench (exact category match).

Usage:
  python scripts/lighting/build_lighting_dataset.py
  python scripts/lighting/build_lighting_dataset.py --dim lighting_type="lighting type"
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from scripts.paths import (
    CSV_DIR,
    SHOTBENCH_DIR,
    SHOTQA_IMAGES_DIR,
    SHOTQA_META_DIR,
    SPLITS_DIR,
    TXT_DIR,
    ensure_output_dirs,
)

DEFAULT_DIMS = {"lighting_type": "lighting type", "lighting_condition": "lighting condition"}

stem = lambda p: os.path.splitext(os.path.basename(p))[0]
norm = lambda s: re.sub(r"\s+", " ", str(s).lower().replace("-", " ")).strip()


def parse_dims(specs: list[str] | None) -> dict[str, str]:
    if not specs:
        return DEFAULT_DIMS
    dims = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--dim must be NAME=category, got {spec!r}")
        name, key = spec.split("=", 1)
        dims[name.strip()] = key.strip().lower()
    return dims


def load_meta() -> pd.DataFrame:
    rows = [json.loads(line) for line in open(SHOTQA_META_DIR / "meta.jsonl")]
    meta = pd.DataFrame(
        [(stem(d.get("img_id") or ""), d.get("title") or "") for d in rows],
        columns=["stem", "title"],
    ).drop_duplicates("stem")
    meta["film"] = meta["title"].str.split("/").str[0].str.strip()
    return meta


def collect_train(key: str, lab: pd.DataFrame, grpo: list, meta: pd.DataFrame) -> pd.DataFrame:
    sft = lab[lab["question"].str.lower().str.contains(key, regex=False)][["image", "label"]].copy()
    sft["source"] = "sft"
    grp = pd.DataFrame(
        [
            (x["images"][0], ", ".join(x["_labels"]), "grpo")
            for x in grpo
            if key in str(x.get("_category", "")).lower()
        ],
        columns=["image", "label", "source"],
    )
    df = pd.concat([sft, grp]).drop_duplicates("image")
    df["stem"] = df["image"].apply(stem)
    df = df.merge(meta, on="stem", how="left")
    df["multi"] = df["label"].str.contains(",")
    return df.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dim", action="append", metavar="NAME=category",
                        help="repeatable; default: lighting_type and lighting_condition")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dims = parse_dims(args.dim)

    ensure_output_dirs()
    meta = load_meta()
    lab = pd.read_csv(CSV_DIR / "shotqa_labels.csv")
    grpo = json.load(open(SHOTQA_META_DIR / "grpo.json"))

    ann = pd.read_csv(CSV_DIR / "annotations.csv")
    ann["img"] = ann["path"].apply(lambda s: ast.literal_eval(s)[0])
    ann["label"] = [json.loads(o)[a] for o, a in zip(ann["options"], ann["answer"])]
    ann["stem"] = ann["img"].apply(stem)
    ann["path"] = ann["img"].apply(lambda p: os.path.join(SHOTBENCH_DIR, p))
    ann = ann.merge(meta[["stem", "film"]], on="stem", how="left")
    films_test = set(ann["film"].dropna())

    extract_list: set[str] = set()
    for name, key in dims.items():
        print(f"\n=== {name} (category: {key!r}) ===")
        key = key.lower()

        # ---- test (ShotBench)
        test = ann[ann["category"].str.lower() == key].copy()
        if test.empty:
            print(f"WARNING: no ShotBench rows with category == {key!r}; skipping {name}. "
                  "Run discover_lighting.py to see the real names.")
            continue
        test["label"] = test["label"].apply(norm)
        test["multi"] = test["label"].str.contains(",")
        exists = test["path"].apply(os.path.exists)
        print(f"test: {len(test)} | multi-label: {int(test['multi'].sum())} | missing on disk: {int((~exists).sum())}")
        test = test[exists][["path", "label", "multi", "film"]]
        test.to_csv(SPLITS_DIR / f"test_{name}.csv", index=False)

        # ---- train/val (ShotQA)
        df = collect_train(key, lab, grpo, meta)
        if df.empty:
            print(f"WARNING: no ShotQA rows for {key!r}; skipping {name}.")
            continue
        df.to_csv(CSV_DIR / f"labels_{name}.csv", index=False)
        extract_list |= {"./" + os.path.basename(p) for p in df["image"]}
        print(f"ShotQA: {len(df)} images | {int((~df['multi']).sum())} single-label | "
              f"{df['film'].nunique()} films | without film meta: {int(df['film'].isna().sum())}")

        df = df[~df["multi"]].copy()
        df["label"] = df["label"].apply(norm)
        df["path"] = df["image"].apply(lambda p: os.path.join(SHOTQA_IMAGES_DIR, os.path.basename(p)))
        n0 = len(df)
        df = df[~df["film"].isin(films_test)]
        print(f"after dropping films present in ShotBench: {n0} -> {len(df)}")
        on_disk = df["path"].apply(os.path.exists)
        if (~on_disk).any():
            print(f"WARNING: {int((~on_disk).sum())} labelled images are not on disk "
                  "(extract them using txt/lista_lighting_extrair.txt); they are excluded.")
        df = df[on_disk].reset_index(drop=True)
        if len(df) < 10:
            print(f"Too few usable training images for {name}; skipping split generation.")
            continue

        print("classes only in train:", set(df["label"]) - set(test["label"]),
              "| only in test:", set(test["label"]) - set(df["label"]))

        groups = df["film"].fillna(df["stem"])
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        tr, va = next(splitter.split(df, groups=groups))
        df.iloc[tr][["path", "label", "film"]].to_csv(SPLITS_DIR / f"train_{name}.csv", index=False)
        df.iloc[va][["path", "label", "film"]].to_csv(SPLITS_DIR / f"val_{name}.csv", index=False)
        print("train / val / test:", len(tr), "/", len(va), "/", len(test))
        print(pd.DataFrame({
            "train": df.iloc[tr]["label"].value_counts(),
            "val": df.iloc[va]["label"].value_counts(),
            "test": test["label"].value_counts(),
        }).fillna(0).astype(int))

    (TXT_DIR / "lista_lighting_extrair.txt").write_text("\n".join(sorted(extract_list)) + "\n")
    print(f"\nWrote {TXT_DIR / 'lista_lighting_extrair.txt'} ({len(extract_list)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
