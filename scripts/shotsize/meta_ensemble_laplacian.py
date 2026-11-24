import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.paths import DATA_PROCESSED_DIR, ensemble_artifacts, split_file


METRICS_PATH, PREDICTIONS_DIR, PROBS_DIR = ensemble_artifacts()
PROCESSED_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR / "meta_ensemble_laplacian"

ensemble_metrics = json.loads(
    METRICS_PATH.read_text(encoding="utf-8")
)

classes = ensemble_metrics["classes"]
class_to_id = {label: i for i, label in enumerate(classes)}


def load_split(split: str):
    features = pd.read_csv(PROCESSED_DIR / f"shot_features_{split}.csv")

    ensemble_probs = np.load(
        PROBS_DIR / f"probs_{split}_plain.npy"
    )

    predictions = pd.read_csv(
        PREDICTIONS_DIR / f"predictions_{split}_plain.csv"
    )

    if len(features) != len(ensemble_probs):
        raise ValueError(
            f"{split}: {len(features)} feature rows, "
            f"but {len(ensemble_probs)} ensemble rows"
        )

    if not features["path"].equals(predictions["path"]):
        raise ValueError(f"{split}: path order differs between files")

    single = ~features["label"].str.contains(",", na=False)

    features = features.loc[single].reset_index(drop=True)
    predictions = predictions.loc[single].reset_index(drop=True)
    ensemble_probs = ensemble_probs[single]

    y = features["label"].map(class_to_id).to_numpy()

    if split == "val":
        split_frame = pd.read_csv(split_file("val_shot_size.csv"))

        if "film" not in split_frame.columns:
            raise ValueError("val_shot_size.csv is missing the film column")

        film_by_path = split_frame.set_index("path")["film"].astype(str)
        groups = (
            features["path"]
            .map(film_by_path)
            .fillna("unknown")
            .to_numpy()
        )
    else:
        groups = np.full(len(features), "unknown", dtype=object)

    laplacian = features[
        ["laplacian_sharpness", "laplacian_center"]
    ].clip(lower=0)

    laplacian = np.log1p(laplacian.to_numpy())

    X = np.concatenate([ensemble_probs, laplacian], axis=1)
    return X, y, groups, predictions


def scores(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=list(range(len(classes))),
                zero_division=0,
            )
        ),
    }


X_val, y_val, val_groups, val_predictions = load_split("val")
X_test, y_test, test_groups, test_predictions = load_split("test")

baseline_val_predictions = (
    val_predictions["prediction"].map(class_to_id).to_numpy()
)
baseline_val_scores = scores(y_val, baseline_val_predictions)

print("Baseline ensemble validation:", baseline_val_scores)

cv = GroupKFold(n_splits=5)

best_c = None
best_cv_f1 = -float("inf")

for c in (0.01, 0.1, 1.0, 10.0):
    fold_scores = []

    for train_idx, fold_idx in cv.split(
        X_val,
        y_val,
        groups=val_groups,
    ):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=c,
                max_iter=3000,
                class_weight="balanced",
            ),
        )

        model.fit(X_val[train_idx], y_val[train_idx])
        predicted = model.predict(X_val[fold_idx])

        fold_scores.append(
            f1_score(
                y_val[fold_idx],
                predicted,
                average="macro",
                labels=list(range(len(classes))),
                zero_division=0,
            )
        )

    mean_f1 = float(np.mean(fold_scores))
    print(f"C={c:g}: grouped-CV val macro-F1={mean_f1:.4f}")

    if mean_f1 > best_cv_f1:
        best_cv_f1 = mean_f1
        best_c = c

print(f"Selected C={best_c:g}; grouped-CV F1={best_cv_f1:.4f}")

final_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(
        C=best_c,
        max_iter=3000,
        class_weight="balanced",
    ),
)

# Trained on validation: test is evaluated only once.
final_model.fit(X_val, y_val)
hybrid_predictions = final_model.predict(X_test)

print(
    "Hybrid ensemble + Laplacian test:",
    scores(y_test, hybrid_predictions),
)

baseline_test_predictions = (
    test_predictions["prediction"].map(class_to_id).to_numpy()
)
print(
    "Original ensemble test:",
    scores(y_test, baseline_test_predictions),
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_DIR / "y_test.npy", y_test)
np.save(
    OUTPUT_DIR / "ensemble_test_predictions.npy",
    baseline_test_predictions,
)
np.save(
    OUTPUT_DIR / "laplacian_test_predictions.npy",
    hybrid_predictions,
)