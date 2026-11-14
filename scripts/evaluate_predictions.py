"""Score prediction CSVs from different systems with ONE metric implementation.

Each system is one or more prediction CSVs (several files = several seeds).
A CSV needs the columns `path`, `label` and `prediction`; `film` (for cluster
bootstrap) and `multi` are used when present. Both compare_shotsize_models.py
and finetune_shotsize.py already write this format.

Example:
  python evaluate_predictions.py \\
    --system svm=data/processed/compare_shotsize/predictions/base_patch16_224__svm_rbf_test.csv \\
    --system finetune="data/processed/finetune/seed*/predictions_test.csv" \\
    --baseline svm \\
    --class-mapping data/processed/finetune/seed42/class_mapping.json \\
    --class-order "extreme close-up" "close-up" ... "extreme long shot"

What it reports
  * accuracy, macro-F1 (classes present in the evaluated rows), macro-F1 over
    all learnable classes, and optionally accuracy within +-1 class;
  * per system: mean and std over its files (seeds), plus a 95% bootstrap CI of
    the seed-averaged metric;
  * paired differences against --baseline, using the SAME resampled test sets
    for every system;
  * summed confusion matrices.

The bootstrap resamples whole films when a `film` column exists (frames of a
film are correlated), otherwise images, and then the CIs are too narrow.
It captures test-set sampling noise; training-seed noise is shown by the std.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRUTHY = {"true", "1", "yes"}
BASE_METRICS = ["accuracy", "macro_f1", "macro_f1_all_learnable"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=PATH_OR_GLOB",
        help="a system and its prediction CSV(s); repeat for several systems",
    )
    p.add_argument("--baseline", default=None, help="system to compare against (default: first)")
    p.add_argument(
        "--class-mapping",
        type=Path,
        default=None,
        help="class_mapping.json written by finetune_shotsize.py (keys = learnable classes). "
        "Default: every label that any system predicts.",
    )
    p.add_argument(
        "--class-order",
        nargs="+",
        default=None,
        help="classes from tightest to widest shot; enables accuracy within +-1 class",
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    return p.parse_args()


def parse_systems(specs: list[str]) -> dict[str, list[Path]]:
    systems: dict[str, list[Path]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--system must be NAME=PATH_OR_GLOB, got {spec!r}")
        name, pattern = spec.split("=", 1)
        files = sorted(Path(f) for f in glob.glob(pattern))
        if not files:
            raise SystemExit(f"No files match {pattern!r} for system {name!r}")
        if name in systems:
            raise SystemExit(f"Duplicate system name {name!r}")
        systems[name] = files
    return systems


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"path", "label", "prediction"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns {sorted(missing)}")
    if frame["path"].duplicated().any():
        raise SystemExit(f"{path} has duplicate paths")
    frame["label"] = frame["label"].astype(str).str.strip()
    frame["prediction"] = frame["prediction"].astype(str).str.strip()
    return frame.set_index("path")


def align(systems: dict[str, list[Path]]) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    """Return the reference frame and, per system, predictions aligned to it."""
    first_file = next(iter(systems.values()))[0]
    ref = read_predictions(first_file)
    aligned: dict[str, list[np.ndarray]] = {}
    for name, files in systems.items():
        aligned[name] = []
        for file in files:
            frame = read_predictions(file)
            if set(frame.index) != set(ref.index):
                only_ref = len(set(ref.index) - set(frame.index))
                only_new = len(set(frame.index) - set(ref.index))
                raise SystemExit(
                    f"{file} covers different images than {first_file} "
                    f"({only_ref} missing, {only_new} extra). All systems must be "
                    "evaluated on the same test rows."
                )
            frame = frame.loc[ref.index]
            if not (frame["label"].to_numpy() == ref["label"].to_numpy()).all():
                raise SystemExit(f"{file} has different labels than {first_file} for the same paths")
            aligned[name].append(frame["prediction"].to_numpy())
    return ref, aligned


def fast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    rank: np.ndarray | None,
) -> np.ndarray:
    """accuracy, macro-F1 over present classes, macro-F1 over all classes[, within-1]."""
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    denom = support + predicted
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    present = support > 0
    out = [
        tp.sum() / len(y_true),
        f1[present].mean() if present.any() else 0.0,
        f1.mean(),
    ]
    if rank is not None:
        out.append(float(np.mean(np.abs(rank[y_true] - rank[y_pred]) <= 1)))
    return np.asarray(out)


def make_resamples(
    n: int, clusters: np.ndarray | None, n_boot: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if clusters is None:
        return [rng.integers(0, n, n) for _ in range(n_boot)]
    groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    resamples = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(groups), len(groups))
        resamples.append(np.concatenate([groups[i] for i in picked]))
    return resamples


def ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, q)) for q in (0.025, 0.975)]


def main() -> None:
    args = parse_args()
    systems = parse_systems(args.system)
    baseline = args.baseline or next(iter(systems))
    if baseline not in systems:
        raise SystemExit(f"--baseline {baseline!r} is not one of {list(systems)}")

    ref, aligned = align(systems)

    # Learnable classes -------------------------------------------------------
    if args.class_mapping is not None:
        learnable = sorted(json.loads(args.class_mapping.read_text(encoding="utf-8")))
    else:
        learnable = sorted({p for preds in aligned.values() for arr in preds for p in set(arr)})
        print("NOTE: no --class-mapping given; learnable classes = every predicted label.")
    classes = learnable
    k = len(classes)
    class_id = {c: i for i, c in enumerate(classes)}

    rank = None
    if args.class_order is not None:
        if set(args.class_order) != set(classes):
            raise SystemExit(
                f"--class-order {args.class_order} does not match the learnable classes {classes}"
            )
        rank = np.zeros(k, dtype=int)
        for r, c in enumerate(args.class_order):
            rank[class_id[c]] = r
    metric_names = BASE_METRICS + (["within_1_class"] if rank is not None else [])

    # Evaluated rows ----------------------------------------------------------
    labels = ref["label"].to_numpy()
    multi = pd.Series(np.char.find(labels.astype(str), ",") >= 0, index=ref.index)
    if "multi" in ref.columns:
        multi |= ref["multi"].astype(str).str.strip().str.lower().isin(TRUTHY)
    known = ref["label"].isin(classes)
    strict = (~multi & known).to_numpy()
    if not strict.any():
        raise SystemExit("No evaluable rows (single label, class seen in training).")
    print(
        f"Rows: {len(ref)} total | {int(strict.sum())} evaluated | "
        f"{int(multi.sum())} multi-label | {int((~multi & ~known).sum())} with unseen class"
    )

    y_true = np.array([class_id[v] for v in labels[strict]])
    y_preds: dict[str, list[np.ndarray]] = {}
    for name, preds in aligned.items():
        y_preds[name] = []
        for arr in preds:
            sel = arr[strict]
            unknown = set(sel) - set(classes)
            if unknown:
                raise SystemExit(f"System {name!r} predicts labels outside the class list: {sorted(unknown)}")
            y_preds[name].append(np.array([class_id[v] for v in sel]))

    clusters = None
    if "film" in ref.columns:
        film = np.array(
            [str(f) if pd.notna(f) else f"__nofilm_{i}" for i, f in enumerate(ref["film"].to_numpy(dtype=object))]
        )
        clusters = film[strict]
        print(f"Bootstrap unit: film ({len(np.unique(clusters))} films)")
    else:
        print("WARNING: no 'film' column; bootstrapping images, so CIs will be too narrow.")

    # Point estimates per file -----------------------------------------------
    point = {
        name: np.array([fast_metrics(y_true, yp, k, rank) for yp in preds])
        for name, preds in y_preds.items()
    }

    # Bootstrap (same resamples for every system -> paired) -------------------
    resamples = make_resamples(len(y_true), clusters, args.bootstrap, args.seed)
    boot = {name: np.zeros((args.bootstrap, len(metric_names))) for name in y_preds}
    for b, idx in enumerate(resamples):
        yt = y_true[idx]
        for name, preds in y_preds.items():
            boot[name][b] = np.mean([fast_metrics(yt, yp[idx], k, rank) for yp in preds], axis=0)

    # Report ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "evaluated_rows": int(strict.sum()),
        "bootstrap": {
            "resamples": args.bootstrap,
            "seed": args.seed,
            "unit": "film" if clusters is not None else "image",
        },
        "classes": classes,
        "systems": {},
        "paired_vs_baseline": {"baseline": baseline, "comparisons": {}},
    }
    rows = []
    for name in systems:
        pts = point[name]
        entry = {"files": [str(f) for f in systems[name]], "n_files": len(pts), "metrics": {}}
        for j, metric in enumerate(metric_names):
            mean = float(pts[:, j].mean())
            std = float(pts[:, j].std(ddof=1)) if len(pts) > 1 else None
            lo, hi = ci(boot[name][:, j])
            entry["metrics"][metric] = {
                "mean": mean,
                "std_over_files": std,
                "ci95_bootstrap": [lo, hi],
                "per_file": [float(v) for v in pts[:, j]],
            }
            rows.append(
                {
                    "system": name,
                    "n_files": len(pts),
                    "metric": metric,
                    "mean": mean,
                    "std_over_files": std,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
        cm = sum(np.bincount(y_true * k + yp, minlength=k * k).reshape(k, k) for yp in y_preds[name])
        entry["confusion_matrix_summed_over_files"] = {
            "labels": classes,
            "rows_true_cols_pred": cm.tolist(),
        }
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(args.output_dir / f"confusion_{name}.csv")
        results["systems"][name] = entry

    pair_rows = []
    for name in systems:
        if name == baseline:
            continue
        comp = {}
        for j, metric in enumerate(metric_names):
            diff_point = float(point[name][:, j].mean() - point[baseline][:, j].mean())
            diff_boot = boot[name][:, j] - boot[baseline][:, j]
            lo, hi = ci(diff_boot)
            p_not_better = float(np.mean(diff_boot <= 0))
            comp[metric] = {
                "difference": diff_point,
                "ci95": [lo, hi],
                "share_of_resamples_not_better": p_not_better,
            }
            pair_rows.append(
                {
                    "system": name,
                    "baseline": baseline,
                    "metric": metric,
                    "difference": diff_point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "share_not_better": p_not_better,
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                }
            )
        results["paired_vs_baseline"]["comparisons"][name] = comp

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "paired_differences.csv", index=False)
    (args.output_dir / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    pd.set_option("display.width", 160)
    print("\n=== Per system (mean over files; CI = bootstrap of the file-averaged metric) ===")
    shown = summary.copy()
    for col in ("mean", "std_over_files", "ci95_low", "ci95_high"):
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    print(shown.to_string(index=False))
    if pair_rows:
        print(f"\n=== Paired difference vs baseline '{baseline}' (positive = better than baseline) ===")
        pr = pd.DataFrame(pair_rows)
        for col in ("difference", "ci95_low", "ci95_high", "share_not_better"):
            pr[col] = pr[col].map(lambda v: f"{v:+.3f}" if col != "share_not_better" else f"{v:.3f}")
        print(pr.drop(columns=["baseline"]).to_string(index=False))
    print(f"\nSaved to {args.output_dir}/ (summary.csv, paired_differences.csv, evaluation.json, confusion_*.csv)")


if __name__ == "__main__":
    main()