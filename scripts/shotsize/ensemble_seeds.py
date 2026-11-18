"""Ensemble several fine-tuned runs (seeds) by averaging softmax probabilities.

For every run directory the script uses logits_val.npy / logits_test.npy when
they exist. Otherwise (for example runs made before finetune_shotsize.py saved
logits) it loads best_model.pt, runs inference only, and saves the logits next
to the checkpoint. With --tta it also scores horizontally flipped images
(logits_*_flip.npy) and averages both views.

Selection must use the VALIDATION columns of the printed table. The test
columns are reported for completeness (look at them once, for the final
configuration), and the saved predictions_*.csv files are compatible with
evaluate_predictions.py.

Example:
  python scripts/shotsize/ensemble_seeds.py \\
    --runs "data/processed/finetune_shotsize_protocol/seed*" --tta
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_RUNS = "data/processed/finetune_shotsize_protocol/seed*"
SPLITS = ("val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs", nargs="+", default=[DEFAULT_RUNS], help="run directories or globs")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/finetune_shotsize_protocol/ensemble"),
    )
    p.add_argument("--tta", action="store_true", help="also use horizontally flipped images")
    p.add_argument("--recompute", action="store_true", help="recompute logits even if present")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_frame(run_dir: Path, split: str) -> pd.DataFrame:
    csv = run_dir / f"predictions_{split}.csv"
    if csv.exists():
        frame = pd.read_csv(csv)
    else:
        import scripts.shotsize.finetune_shotsize as fs  # needs torch; only for runs without saved CSVs

        frame = fs.load_split(split)
    frame = frame.drop(columns=["prediction"], errors="ignore")
    frame["label"] = frame["label"].astype(str).str.strip()
    return frame


def compute_missing_logits(run_dir: Path, wanted: list[str], args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    import scripts.shotsize.finetune_shotsize as fs

    device = fs.select_device()
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    label_to_id = checkpoint["label_to_id"]
    backbone = AutoModel.from_pretrained(checkpoint["model_name"])
    feature_dim = getattr(backbone.config, "projection_dim", None)
    if feature_dim is None:
        feature_dim = backbone.config.vision_config.hidden_size
    model = fs.SigLIPShotSizeClassifier(backbone, feature_dim, len(label_to_id))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    processor = AutoImageProcessor.from_pretrained(checkpoint["model_name"])

    for key in wanted:
        flip = key.endswith("_flip")
        split = key.removesuffix("_flip")
        frame = fs.load_split(split)
        predictions, logits = fs.predict(
            model,
            frame,
            label_to_id,
            processor,
            device,
            args.batch_size,
            args.num_workers,
            flip=flip,
        )
        np.save(run_dir / f"logits_{key}.npy", logits)
        saved = run_dir / f"predictions_{split}.csv"
        if not flip and saved.exists():
            old = pd.read_csv(saved)
            if len(old) == len(predictions):
                same = float((old["prediction"].to_numpy() == predictions["prediction"].to_numpy()).mean())
                print(f"  {run_dir.name}/{key}: recomputed predictions match the saved CSV on {same:.1%} of rows")


def single_label_scores(
    labels: np.ndarray, probs: np.ndarray, classes: list[str]
) -> tuple[float, float]:
    keep = np.array([("," not in label) and (label in classes) for label in labels])
    y_true = labels[keep]
    y_pred = np.array(classes)[probs.argmax(axis=1)][keep]
    present = sorted(set(y_true))
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
    return acc, f1


def main() -> None:
    args = parse_args()
    runs = sorted({Path(p) for pattern in args.runs for p in glob.glob(pattern) if Path(p).is_dir()})
    if not runs:
        raise SystemExit(f"No run directories match {args.runs}")
    print(f"{len(runs)} run(s): {', '.join(r.name for r in runs)}")

    mappings = [json.loads((r / "class_mapping.json").read_text(encoding="utf-8")) for r in runs]
    if any(m != mappings[0] for m in mappings):
        raise SystemExit("The runs do not share the same class_mapping.json")
    classes = [c for c, _ in sorted(mappings[0].items(), key=lambda item: item[1])]

    needed = list(SPLITS) + ([f"{s}_flip" for s in SPLITS] if args.tta else [])
    logits: dict[Path, dict[str, np.ndarray]] = {}
    for run in runs:
        missing = [k for k in needed if args.recompute or not (run / f"logits_{k}.npy").exists()]
        if missing:
            print(f"{run.name}: computing {missing} from best_model.pt")
            compute_missing_logits(run, missing, args)
        logits[run] = {k: np.load(run / f"logits_{k}.npy") for k in needed}

    frames = {split: load_frame(runs[0], split) for split in SPLITS}
    for split in SPLITS:
        paths = frames[split]["path"].tolist()
        for run in runs:
            if len(logits[run][split]) != len(paths):
                raise SystemExit(
                    f"{run.name}: logits_{split}.npy has {len(logits[run][split])} rows, "
                    f"the {split} split has {len(paths)}"
                )
            other = run / f"predictions_{split}.csv"
            if other.exists() and pd.read_csv(other)["path"].tolist() != paths:
                raise SystemExit(f"{run.name}: {split} rows are in a different order than {runs[0].name}")

    variants = {"plain": [""]}
    if args.tta:
        variants["tta"] = ["", "_flip"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary: dict = {"runs": [str(r) for r in runs], "classes": classes, "variants": {}}
    for variant, suffixes in variants.items():
        per_run_probs = {
            split: [
                np.mean([softmax(logits[run][f"{split}{s}"]) for s in suffixes], axis=0)
                for run in runs
            ]
            for split in SPLITS
        }
        entry: dict = {}
        for split in SPLITS:
            labels = frames[split]["label"].to_numpy()
            individual = np.array([single_label_scores(labels, p, classes) for p in per_run_probs[split]])
            ensemble_probs = np.mean(per_run_probs[split], axis=0)
            ens_acc, ens_f1 = single_label_scores(labels, ensemble_probs, classes)
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "individual_acc_mean": individual[:, 0].mean(),
                    "individual_f1_mean": individual[:, 1].mean(),
                    "individual_f1_std": individual[:, 1].std(ddof=1) if len(runs) > 1 else np.nan,
                    "ensemble_acc": ens_acc,
                    "ensemble_f1": ens_f1,
                    "ensemble_gain_f1": ens_f1 - individual[:, 1].mean(),
                }
            )
            entry[split] = {
                "individual_f1": [float(v) for v in individual[:, 1]],
                "individual_acc": [float(v) for v in individual[:, 0]],
                "ensemble_acc": ens_acc,
                "ensemble_f1": ens_f1,
            }
            predictions = frames[split].copy()
            predictions["prediction"] = np.array(classes)[ensemble_probs.argmax(axis=1)]
            predictions.to_csv(args.output_dir / f"predictions_{split}_{variant}.csv", index=False)
            np.save(args.output_dir / f"probs_{split}_{variant}.npy", ensemble_probs)
        summary["variants"][variant] = entry

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "ensemble_summary.csv", index=False)
    (args.output_dir / "ensemble_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.set_option("display.width", 160)
    shown = table.copy()
    for col in shown.columns[2:]:
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    print("\n=== Individual runs vs ensemble (single-label rows) ===")
    print(shown.to_string(index=False))
    print(
        "\nDecide with the 'val' rows. Use the 'test' rows only for the final configuration.\n"
        f"Saved to {args.output_dir}/ (predictions_<split>_<variant>.csv work with evaluate_predictions.py)"
    )


if __name__ == "__main__":
    main()