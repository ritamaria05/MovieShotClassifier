"""Evaluate lighting classifiers on frozen SigLIP embeddings, handcrafted
lighting features, or both.

Feature sets (--feature-sets):
  embeddings   L2-normalised SigLIP image embeddings
  handcrafted  descriptors from extract_lighting_features.py (standardised)
  hybrid       both concatenated (standardised)
Protocol (same as embed_framing.py): choose hyperparameters by validation
macro-F1 on train only, refit on train+val, then evaluate once on the test set.
Strict metrics use single-label test rows whose class occurs in train+val;
lenient accuracy counts a prediction correct if it matches any recorded label.
A majority-class baseline is always reported.

Embeddings are cached at data/processed/emb/<split>_<dim><TAG>.npy and are never
recomputed when the cache exists (row count is verified against the split CSV).

Usage:
  python scripts/lighting/embed_lighting.py --dim lighting_type
  python scripts/lighting/embed_lighting.py --dim lighting_condition --classifier mlp
  python scripts/lighting/embed_lighting.py --dim lighting_type --feature-sets handcrafted
Outputs go to outputs/lighting/ (metrics JSON, confusion matrix, predictions).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.framing.mlp_head import MLP_GRID, make_mlp
from scripts.paths import DATA_PROCESSED_DIR, SPLITS_DIR

DEFAULT_MODEL = "google/siglip-base-patch16-224"
EMB_DIR = DATA_PROCESSED_DIR / "emb"
HAND_DIR = DATA_PROCESSED_DIR / "lighting"
SPLIT_NAMES = ("train", "val", "test")
NEAR_DUPLICATE = 0.95


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dim", default="lighting_type", help="split name stem, e.g. lighting_type")
    p.add_argument("--classifier", choices=["logreg", "mlp"], default="logreg")
    p.add_argument("--feature-sets", nargs="+", default=["embeddings", "handcrafted", "hybrid"],
                   choices=["embeddings", "handcrafted", "hybrid"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "lighting")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


# ---------------------------------------------------------------- data
def load_splits(dim: str) -> dict[str, pd.DataFrame]:
    out = {}
    for split in SPLIT_NAMES:
        path = SPLITS_DIR / f"{split}_{dim}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run build_lighting_dataset.py first.")
        frame = pd.read_csv(path)
        frame["label"] = frame["label"].astype(str).str.strip()
        if "multi" not in frame.columns:
            frame["multi"] = frame["label"].str.contains(",", regex=False)
        out[split] = frame
    return out


def load_image(path: str) -> Image.Image:
    image = Image.open(path)
    image.draft("RGB", (448, 448))  # decode JPEGs at reduced size
    return image.convert("RGB")


def embedding_tag(model_name: str) -> str:
    return "" if model_name == DEFAULT_MODEL else "_" + model_name.split("/")[-1]


def load_embeddings(dim: str, splits: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, np.ndarray]:
    import torch  # imported lazily so handcrafted-only runs need no torch

    tag = embedding_tag(args.model)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    model = processor = None
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    result = {}
    for split, frame in splits.items():
        cache = EMB_DIR / f"{split}_{dim}{tag}.npy"
        if cache.exists():
            emb = np.load(cache)
            if emb.ndim != 2 or len(emb) != len(frame):
                raise ValueError(f"{cache} has shape {emb.shape}, split has {len(frame)} rows; delete the cache.")
            result[split] = emb
            continue
        if model is None:
            from transformers import AutoImageProcessor, AutoModel
            model = AutoModel.from_pretrained(args.model).eval().to(device)
            processor = AutoImageProcessor.from_pretrained(args.model)
        from tqdm import tqdm
        paths, chunks = frame["path"].tolist(), []
        with torch.no_grad(), ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i in tqdm(range(0, len(paths), args.batch_size), desc=cache.name):
                images = list(pool.map(load_image, paths[i: i + args.batch_size]))
                inputs = processor(images=images, return_tensors="pt").to(device)
                feats = model.get_image_features(**inputs)
                feats = feats if torch.is_tensor(feats) else feats.pooler_output
                chunks.append(feats.float().cpu())
        emb = torch.nn.functional.normalize(torch.cat(chunks), dim=-1).numpy()
        np.save(cache, emb)
        result[split] = emb
    return result


def load_handcrafted(dim: str, splits: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    result = {}
    for split, frame in splits.items():
        path = HAND_DIR / f"features_{dim}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run extract_lighting_features.py first.")
        table = pd.read_csv(path)
        if table["path"].tolist() != frame["path"].tolist():
            raise ValueError(f"{path}: rows differ from the split CSV (order or content). Re-extract.")
        columns = [c for c in table.columns if c not in frame.columns]
        result[split] = np.nan_to_num(table[columns].to_numpy(dtype=float))
    return result


# ---------------------------------------------------------------- models
def classifier_factory(kind: str, scaled: bool):
    if kind == "logreg":
        grid = [{"C": c} for c in (0.01, 0.1, 1, 10, 100)]

        def make(**params):
            return LogisticRegression(max_iter=3000, class_weight="balanced", **params)
    else:
        grid, make = MLP_GRID, make_mlp
    if not scaled:
        return make, grid
    return (lambda **params: make_pipeline(StandardScaler(), make(**params))), grid


def jsonable(params: dict) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}


def evaluate_set(name, X, splits, args, near_dup_mask=None) -> dict:
    tr, va, te = (splits[s] for s in SPLIT_NAMES)
    Xtr, Xva, Xte = (X[s] for s in SPLIT_NAMES)
    make, grid = classifier_factory(args.classifier, scaled=name != "embeddings")

    scores = []
    for params in grid:
        clf = make(**params).fit(Xtr, tr["label"])
        score = float(f1_score(va["label"], clf.predict(Xva), average="macro"))
        scores.append(score)
        print(f"[{name}] {jsonable(params)}: val macro-F1 {score:.3f}", flush=True)
    best = int(np.argmax(scores))
    best_params = grid[best]

    y_fit = pd.concat([tr["label"], va["label"]], ignore_index=True)
    clf = make(**best_params).fit(np.vstack([Xtr, Xva]), y_fit)
    pred = clf.predict(Xte)

    metrics = score_test(te, pred, set(y_fit))
    metrics.update(feature_set=name, classifier=args.classifier, best_params=jsonable(best_params),
                   validation_macro_f1=scores[best],
                   validation_grid=[{"params": jsonable(p), "macro_f1": s} for p, s in zip(grid, scores)])
    if near_dup_mask is not None:
        metrics["test_near_duplicates"] = int(near_dup_mask.sum())

    stem = f"lighting_{args.dim}_{name}_{args.classifier}"
    args.out.mkdir(parents=True, exist_ok=True)
    true = te["label"].to_numpy()
    single = ~te["multi"].to_numpy()
    known = te["label"].isin(set(y_fit)).to_numpy()
    strict = single & known
    labels = sorted(set(true[strict]) | set(pred[strict]))
    pd.DataFrame(confusion_matrix(true[strict], pred[strict], labels=labels),
                 index=labels, columns=labels).to_csv(args.out / f"{stem}_confusion.csv", index_label="actual")
    out_pred = te.copy()
    out_pred["prediction"] = pred
    out_pred.to_csv(args.out / f"{stem}_predictions.csv", index=False)
    metrics["classification_report"] = classification_report(
        true[strict], pred[strict], labels=labels, output_dict=True, zero_division=0)
    (args.out / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[{name}] TEST strict acc {metrics['test_strict_accuracy']:.3f} | "
          f"macro-F1 {metrics['test_strict_macro_f1']:.3f} | lenient {metrics['test_lenient_accuracy']:.3f} "
          f"(n={metrics['test_n_strict']})", flush=True)
    return metrics


def score_test(te: pd.DataFrame, pred: np.ndarray, learnable: set) -> dict:
    true = te["label"].to_numpy()
    single = ~te["multi"].to_numpy()
    known = te["label"].isin(learnable).to_numpy()
    strict = single & known
    if not strict.any():
        raise ValueError("No strict single-label test rows with a class seen in training.")
    classes = sorted(learnable)
    lenient = np.array([p in {x.strip() for x in label.split(",")} for p, label in zip(pred, true)])
    return {
        "test_strict_accuracy": float(accuracy_score(true[strict], pred[strict])),
        "test_strict_macro_f1": float(f1_score(true[strict], pred[strict], labels=classes,
                                               average="macro", zero_division=0)),
        "test_n_strict": int(strict.sum()),
        "test_lenient_accuracy": float(lenient.mean()),
        "test_excluded_multi_label": int((~single).sum()),
        "test_excluded_unseen_class": int((single & ~known).sum()),
    }


def main() -> int:
    args = parse_args()
    splits = load_splits(args.dim)
    for split, frame in splits.items():
        print(f"{split}: {len(frame)} rows | classes: {frame['label'].nunique()}")

    # majority baseline
    tr, va, te = (splits[s] for s in SPLIT_NAMES)
    y_fit = pd.concat([tr["label"], va["label"]], ignore_index=True)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_fit), 1)), y_fit)
    baseline = score_test(te, dummy.predict(np.zeros((len(te), 1))), set(y_fit))
    print(f"[majority] strict acc {baseline['test_strict_accuracy']:.3f} | "
          f"macro-F1 {baseline['test_strict_macro_f1']:.3f}")

    emb = hand = None
    if {"embeddings", "hybrid"} & set(args.feature_sets):
        emb = load_embeddings(args.dim, splits, args)
    if {"handcrafted", "hybrid"} & set(args.feature_sets):
        hand = load_handcrafted(args.dim, splits)

    near_dup = None
    if emb is not None:
        sim = (emb["test"] @ np.vstack([emb["train"], emb["val"]]).T).max(axis=1)
        near_dup = sim > NEAR_DUPLICATE
        print(f"max test->train/val similarity: median {np.median(sim):.3f} | "
              f"p95 {np.quantile(sim, .95):.3f} | >{NEAR_DUPLICATE}: {int(near_dup.sum())} of {len(te)}")

    matrices = {}
    if "embeddings" in args.feature_sets:
        matrices["embeddings"] = emb
    if "handcrafted" in args.feature_sets:
        matrices["handcrafted"] = hand
    if "hybrid" in args.feature_sets:
        matrices["hybrid"] = {s: np.hstack([emb[s], hand[s]]) for s in SPLIT_NAMES}

    summary = {"dim": args.dim, "classifier": args.classifier, "majority_baseline": baseline, "results": {}}
    for name, X in matrices.items():
        summary["results"][name] = evaluate_set(name, X, splits, args, near_dup if name == "embeddings" else None)

    print("\n=== summary (strict single-label test) ===")
    print(f"{'majority':<12} acc {baseline['test_strict_accuracy']:.3f}  macro-F1 {baseline['test_strict_macro_f1']:.3f}")
    for name, m in summary["results"].items():
        print(f"{name:<12} acc {m['test_strict_accuracy']:.3f}  macro-F1 {m['test_strict_macro_f1']:.3f}  "
              f"(val macro-F1 {m['validation_macro_f1']:.3f})")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"lighting_{args.dim}_{args.classifier}_summary.json").write_text(
        json.dumps({k: v for k, v in summary.items()}, indent=2, default=float), encoding="utf-8")
    print("Test metrics come from a single look at the test set; choose configurations with validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
