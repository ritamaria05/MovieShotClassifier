"""Compare SigLIP embedding backbones with logistic regression and RBF SVM.

The script deliberately uses the existing film-separated shot-size splits. It
selects classifier hyperparameters on validation only, then refits on
train+validation before evaluating strict single-label test rows.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.svm import SVC
    from tqdm import tqdm
    from transformers import AutoImageProcessor, AutoModel
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing dependency {error.name!r}. Install the repository dependencies "
        "with `pip install -r requirements.txt` before running this comparison."
    ) from error

from paths import DATA_PROCESSED_DIR, SPLITS_DIR


DEFAULT_BASE_MODEL = "google/siglip-base-patch16-224"
DEFAULT_SO400M_MODEL = "google/siglip-so400m-patch14-384"
DEFAULT_OUTPUT_DIR = DATA_PROCESSED_DIR / "compare_shotsize"
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--so400m-model", default=DEFAULT_SO400M_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--logistic-c", type=float, nargs="+", default=[0.1, 1, 10, 100])
    parser.add_argument("--svm-c", type=float, nargs="+", default=[0.1, 1, 10, 100])
    parser.add_argument(
        "--svm-gamma",
        nargs="+",
        type=parse_gamma,
        default=["scale", "auto"],
        help="SVM gamma values: scale, auto, or numeric values",
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    return parser.parse_args()


def parse_gamma(value: str) -> str | float:
    if value in {"scale", "auto"}:
        return value
    try:
        gamma = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "gamma must be scale, auto, or a numeric value"
        ) from error
    if gamma <= 0:
        raise argparse.ArgumentTypeError("numeric gamma must be positive")
    return gamma


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")


def load_image(path: str) -> Image.Image:
    try:
        image = Image.open(path)
        image.draft("RGB", (448, 448))
        return image.convert("RGB")
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"Could not read image {path}: {error}") from error


def load_splits() -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for split in SPLIT_NAMES:
        path = SPLITS_DIR / f"{split}_shot_size.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        required = {"path", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        if "multi" not in frame.columns:
            frame["multi"] = frame["label"].astype(str).str.contains(",", regex=False)
        missing_images = [path for path in frame["path"] if not Path(path).exists()]
        if missing_images:
            raise FileNotFoundError(
                f"{path} references {len(missing_images)} missing image(s), "
                f"for example {missing_images[0]}"
            )
        splits[split] = frame.reset_index(drop=True)
    if missing:
        raise FileNotFoundError(
            "Missing split CSV files. Run the existing data preparation pipeline first:\n"
            + "\n".join(missing)
        )
    return splits


@torch.no_grad()
def embed_split(
    frame: pd.DataFrame,
    cache_path: Path,
    processor: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    image_workers: int,
) -> tuple[np.ndarray, bool, float]:
    started = time.perf_counter()
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.ndim != 2 or len(embeddings) != len(frame):
            raise ValueError(
                f"Embedding cache {cache_path} has shape {embeddings.shape}; "
                f"expected ({len(frame)}, feature_dim). Delete it and retry."
            )
        return embeddings, True, time.perf_counter() - started

    paths = frame["path"].tolist()
    batches: list[torch.Tensor] = []
    with ThreadPoolExecutor(max_workers=image_workers) as executor:
        for start in tqdm(
            range(0, len(paths), batch_size),
            desc=cache_path.name,
            unit="batch",
        ):
            images = list(executor.map(load_image, paths[start : start + batch_size]))
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
            if not torch.is_tensor(features):
                features = features.pooler_output
            batches.append(features.float().cpu())
    if not batches:
        raise ValueError(f"Cannot embed an empty split: {cache_path}")
    embeddings = torch.nn.functional.normalize(torch.cat(batches), dim=-1).numpy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings, False, time.perf_counter() - started


def select_classifier(
    classifier_name: str,
    x_train: np.ndarray,
    y_train: pd.Series,
    x_val: np.ndarray,
    y_val: pd.Series,
    logistic_c: list[float],
    svm_c: list[float],
    svm_gamma: list[str | float],
    max_iter: int,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[dict[str, Any], Any]] = []
    if classifier_name == "logistic":
        for c in logistic_c:
            candidates.append(
                (
                    {"C": c},
                    LogisticRegression(
                        C=c,
                        max_iter=max_iter,
                        class_weight="balanced",
                    ),
                )
            )
    else:
        for c in svm_c:
            for gamma in svm_gamma:
                candidates.append(
                    (
                        {"C": c, "gamma": gamma},
                        SVC(C=c, gamma=gamma, class_weight="balanced", kernel="rbf"),
                    )
                )

    validation_scores: list[dict[str, Any]] = []
    for params, classifier in candidates:
        classifier.fit(x_train, y_train)
        score = float(f1_score(y_val, classifier.predict(x_val), average="macro"))
        validation_scores.append({**params, "macro_f1": score})
    best_index = max(
        range(len(validation_scores)),
        key=lambda index: (validation_scores[index]["macro_f1"], -index),
    )
    best_params = {key: value for key, value in validation_scores[best_index].items() if key != "macro_f1"}
    return candidates[best_index][1], best_params, validation_scores


def evaluate(
    classifier_name: str,
    best_params: dict[str, Any],
    x_train_val: np.ndarray,
    y_train_val: pd.Series,
    x_test: np.ndarray,
    test: pd.DataFrame,
    max_iter: int,
) -> dict[str, Any]:
    if classifier_name == "logistic":
        classifier = LogisticRegression(
            C=best_params["C"], max_iter=max_iter, class_weight="balanced"
        )
    else:
        classifier = SVC(
            C=best_params["C"],
            gamma=best_params["gamma"],
            class_weight="balanced",
            kernel="rbf",
        )
    classifier.fit(x_train_val, y_train_val)
    predictions = classifier.predict(x_test)
    multi = test["multi"]
    if multi.dtype == bool:
        single = ~multi.to_numpy()
    else:
        single = ~multi.astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes"}
        ).to_numpy()
    if not single.any():
        raise ValueError("The test split contains no strict single-label rows.")
    labels = test["label"].to_numpy()
    return {
        "accuracy": float(accuracy_score(labels[single], predictions[single])),
        "macro_f1": float(
            f1_score(labels[single], predictions[single], average="macro")
        ),
        "evaluated_samples": int(single.sum()),
        "excluded_multi_label_examples": int((~single).sum()),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.image_workers < 1:
        raise ValueError("--batch-size and --image-workers must be positive")
    splits = load_splits()
    device = select_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = args.output_dir / "embeddings"
    results: dict[str, Any] = {
        "methodology": {
            "splits": "Existing train_shot_size.csv, val_shot_size.csv, test_shot_size.csv",
            "classifier_tuning": "Validation macro-F1 only",
            "final_fit": "Train plus validation",
            "test_metric": "Strict single-label rows only; multi-label rows excluded",
            "embedding": "L2-normalized SigLIP image features",
        },
        "device": str(device),
        "batch_size": args.batch_size,
        "models": {},
    }

    for model_key, model_name in (
        ("base_patch16_224", args.base_model),
        ("so400m_patch14_384", args.so400m_model),
    ):
        print(f"\n=== {model_key}: {model_name} on {device} ===")
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).eval().to(device)
        except Exception as error:
            raise RuntimeError(
                f"Could not load configured model {model_name!r}; "
                "no fallback model is attempted."
            ) from error

        model_results: dict[str, Any] = {"model_name": model_name, "embeddings": {}}
        features: dict[str, np.ndarray] = {}
        for split in SPLIT_NAMES:
            cache_path = embeddings_dir / f"{model_slug(model_name)}__{split}_shot_size.npy"
            features[split], cached, seconds = embed_split(
                splits[split],
                cache_path,
                processor,
                model,
                device,
                args.batch_size,
                args.image_workers,
            )
            model_results["embeddings"][split] = {
                "path": str(cache_path),
                "cached": cached,
                "seconds": round(seconds, 3),
                "shape": list(features[split].shape),
            }

        y_train, y_val = (splits[split]["label"] for split in ("train", "val"))
        x_train_val = np.vstack([features["train"], features["val"]])
        y_train_val = pd.concat([y_train, y_val], ignore_index=True)
        for classifier_name in ("logistic", "svm_rbf"):
            started = time.perf_counter()
            _, best_params, validation_scores = select_classifier(
                classifier_name,
                features["train"],
                y_train,
                features["val"],
                y_val,
                args.logistic_c,
                args.svm_c,
                args.svm_gamma,
                args.max_iter,
            )
            metrics = evaluate(
                classifier_name,
                best_params,
                x_train_val,
                y_train_val,
                features["test"],
                splits["test"],
                args.max_iter,
            )
            model_results[classifier_name] = {
                "selected_hyperparameters": best_params,
                "validation_scores": validation_scores,
                "test": metrics,
                "seconds": round(time.perf_counter() - started, 3),
            }
        results["models"][model_key] = model_results
        del model, processor
        if device.type == "mps":
            torch.mps.empty_cache()

    result_path = args.output_dir / "comparison_results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved machine-readable results to {result_path}")


if __name__ == "__main__":
    main()
