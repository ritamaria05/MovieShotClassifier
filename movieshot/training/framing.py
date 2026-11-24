"""Fine-tune the base SigLIP vision encoder for framing classification.

Train a head with the backbone frozen, restore its best validation checkpoint,
then unfreeze the last vision layers and fine-tune with a smaller learning rate.
The best checkpoint in each stage is selected by validation macro-F1.

Example:
    python -m scripts.framing.finetune_framing --seeds 42 43 44 --baseline-matrix outputs/csv/confusion_matrix_framing_MLP.csv
    python -m scripts.framing.finetune_framing --benchmark-batches 2

Outputs are separated by seed. The test set is evaluated only after training.
The benchmark runs a few training batches without writing checkpoints or
evaluating validation/test data.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from movieshot.models import FramingClassifier, select_device
from scripts.paths import ROOT, resolve_image_path, split_file


DEFAULT_MODEL = "google/siglip-base-patch16-224"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--unfreeze-last-layers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--benchmark-batches", type=int, default=0)
    parser.add_argument("--baseline-matrix", type=Path, default=None,
                        help="existing strict test confusion matrix for a reference comparison")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(root: Path, name: str) -> pd.DataFrame:
    path = split_file(f"{name}_framing.csv", root)
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"{path}: empty split or missing columns: {sorted(missing)}")
    if frame["path"].isna().any() or frame["label"].isna().any():
        raise ValueError(f"{path}: missing image paths or labels")
    frame["label"] = frame["label"].astype(str).str.strip()
    if frame["label"].eq("").any():
        raise ValueError(f"{path}: empty labels")
    if name != "test" and frame["label"].str.contains(",", regex=False).any():
        raise ValueError(f"{path}: train and validation must have single labels")
    missing_images = [p for p in frame["path"] if not resolve_image_path(p, root).is_file()]
    if missing_images:
        raise FileNotFoundError(f"{path}: {len(missing_images)} missing images: {missing_images[:3]}")
    return frame


class FramingDataset(Dataset):
    def __init__(self, root: Path, frame: pd.DataFrame, label_to_id: dict[str, int]) -> None:
        self.root = root
        self.frame = frame.reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        row = self.frame.iloc[index]
        path = resolve_image_path(row["path"], self.root)
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not read image {path}: {error}") from error
        return rgb, self.label_to_id.get(row["label"], -1), str(row["path"])


def collate(batch: list[tuple[Image.Image, int, str]]) -> tuple[list[Image.Image], torch.Tensor, list[str]]:
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(paths)


def configure_trainable_layers(model: FramingClassifier, layer_count: int) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if layer_count == 0:
        return
    layers = getattr(
        getattr(getattr(model.backbone, "vision_model", None), "encoder", None),
        "layers", None,
    )
    if layer_count < 0 or layers is None or layer_count > len(layers):
        raise ValueError("Could not locate the requested number of SigLIP vision encoder layers")
    for layer in layers[-layer_count:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


def make_loader(root: Path, frame: pd.DataFrame, labels: dict[str, int],
                batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(FramingDataset(root, frame, labels), batch_size=batch_size,
                      shuffle=shuffle, num_workers=workers, collate_fn=collate)


def run_epoch(model: FramingClassifier, loader: DataLoader, processor: Any,
              device: torch.device, optimizer: torch.optim.Optimizer | None,
              max_batches: int = 0) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        # Keep frozen vision layers in evaluation mode while training unfrozen layers.
        model.backbone.eval()
        for module in model.backbone.modules():
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                module.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels, _) in enumerate(tqdm(loader, leave=False)):
            if max_batches and batch_index >= max_batches:
                break
            inputs = {k: v.to(device) for k, v in processor(images=images, return_tensors="pt").items()}
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.cpu().tolist())
    if not targets:
        raise ValueError("No examples were processed")
    return loss_sum / len(targets), float(f1_score(targets, predictions, average="macro"))


def train_stage(model: FramingClassifier, train_loader: DataLoader, val_loader: DataLoader,
                processor: Any, device: torch.device, args: argparse.Namespace,
                stage: str, output: Path) -> dict[str, Any]:
    layer_count = 0 if stage == "head" else args.unfreeze_last_layers
    configure_trainable_layers(model, layer_count)
    groups = [{"params": list(model.classifier.parameters()), "lr": args.head_learning_rate}]
    if layer_count:
        groups.append({"params": [p for p in model.backbone.parameters() if p.requires_grad],
                       "lr": args.backbone_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    epochs = args.head_epochs if stage == "head" else args.finetune_epochs
    history = []
    best_f1, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, processor, device, optimizer)
        val_loss, val_f1 = run_epoch(model, val_loader, processor, device, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                        "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"{stage} epoch {epoch}/{epochs}: train F1 {train_f1:.4f}, "
              f"val F1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"{stage}: early stopping after epoch {epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError(f"{stage}: no valid validation checkpoint")
    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "model_name": args.model_name,
                "hidden_size": args.hidden_size, "stage": stage,
                "epoch": best_epoch, "val_macro_f1": best_f1,
                "label_to_id": args.label_to_id}, output / f"best_{stage}.pt")
    return {"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "history": history}


def predict(model: FramingClassifier, loader: DataLoader, processor: Any,
            device: torch.device, classes: list[str]) -> list[str]:
    model.eval()
    result = []
    with torch.no_grad():
        for images, _, _ in tqdm(loader, desc="predict", leave=False):
            inputs = {k: v.to(device) for k, v in processor(images=images, return_tensors="pt").items()}
            result.extend(classes[i] for i in model(inputs).argmax(dim=1).cpu().tolist())
    return result


def baseline_metrics(path: Path, test: pd.DataFrame) -> dict[str, float]:
    matrix = pd.read_csv(path, index_col="actual")
    true = test.loc[~test["label"].str.contains(",", regex=False), "label"]
    labels = sorted(true.unique())
    if set(matrix.index) != set(labels) or set(matrix.columns) != set(labels):
        raise ValueError(f"{path}: baseline classes do not match test classes")
    matrix = matrix.loc[labels, labels]
    values = matrix.to_numpy()
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError(f"{path}: confusion matrix must contain non-negative counts")
    if not np.array_equal(values.sum(axis=1), true.value_counts().reindex(labels).to_numpy()):
        raise ValueError(f"{path}: baseline supports do not match the test split")
    diagonal = np.diag(values)
    denominators = values.sum(axis=0) + values.sum(axis=1)
    return {"accuracy": float(diagonal.sum() / values.sum()),
            "macro_f1": float(np.mean(2 * diagonal / denominators))}


def run_seed(args: argparse.Namespace, seed: int, splits: dict[str, pd.DataFrame],
             processor: Any, device: torch.device) -> dict[str, Any] | None:
    set_seed(seed)
    classes = sorted(splits["train"]["label"].unique())
    args.label_to_id = {label: i for i, label in enumerate(classes)}
    for name in ("val", "test"):
        unknown = {label for value in splits[name]["label"] for label in value.split(", ")
                   if label not in args.label_to_id}
        if unknown:
            raise ValueError(f"{name} contains labels absent from train: {sorted(unknown)}")
    backbone = AutoModel.from_pretrained(args.model_name)
    feature_dim = backbone.config.vision_config.hidden_size
    model = FramingClassifier(backbone, feature_dim, args.hidden_size, len(classes)).to(device)
    loaders = {name: make_loader(args.root, frame, args.label_to_id, args.batch_size,
                                 args.num_workers, name == "train")
               for name, frame in splits.items()}
    if args.benchmark_batches:
        configure_trainable_layers(model, args.unfreeze_last_layers)
        optimizer = torch.optim.AdamW([
            {"params": model.classifier.parameters(), "lr": args.head_learning_rate},
            {"params": [p for p in model.backbone.parameters() if p.requires_grad],
             "lr": args.backbone_learning_rate},
        ], weight_decay=args.weight_decay)
        start = time.perf_counter()
        run_epoch(model, loaders["train"], processor, device, optimizer, args.benchmark_batches)
        if device.type == "cuda":
            torch.cuda.synchronize()
        count = min(len(splits["train"]), args.benchmark_batches * args.batch_size)
        elapsed = time.perf_counter() - start
        print(f"Benchmark: {count} examples in {elapsed:.1f}s ({elapsed/count:.2f} s/example); "
              f"approx. {elapsed/count*len(splits['train'])/60:.1f} min/train epoch; "
              "excludes validation and model loading.")
        return None

    output = args.output_dir / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    head = train_stage(model, loaders["train"], loaders["val"], processor, device, args, "head", output)
    fine = train_stage(model, loaders["train"], loaders["val"], processor, device, args, "finetune", output)
    test = splits["test"].copy()
    test["prediction"] = predict(model, loaders["test"], processor, device, classes)
    single = ~test["label"].str.contains(",", regex=False)
    if not single.any():
        raise ValueError("test split has no single-label examples")
    strict = test.loc[single]
    lenient = [p in [x.strip() for x in label.split(",")]
               for label, p in zip(test["label"], test["prediction"])]
    matrix_labels = sorted(set(strict["label"]) | set(strict["prediction"]))
    matrix = confusion_matrix(strict["label"], strict["prediction"], labels=matrix_labels)
    pd.DataFrame(matrix, index=matrix_labels, columns=matrix_labels).to_csv(
        output / "confusion_matrix.csv", index_label="actual")
    test.to_csv(output / "predictions_test.csv", index=False)
    metrics = {
        "seed": seed, "model_name": args.model_name,
        "head": head, "finetune": fine,
        "test_strict_accuracy": float(accuracy_score(strict["label"], strict["prediction"])),
        "test_strict_macro_f1": float(f1_score(strict["label"], strict["prediction"], average="macro")),
        "test_n_strict": len(strict), "test_lenient_accuracy": float(np.mean(lenient)),
        "test_classification_report": classification_report(
            strict["label"], strict["prediction"], output_dict=True, zero_division=0),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"seed {seed}: test accuracy {metrics['test_strict_accuracy']:.3f}, "
          f"macro-F1 {metrics['test_strict_macro_f1']:.3f}", flush=True)
    return metrics


def main() -> int:
    args = parse_args()
    if (args.head_epochs < 1 or args.finetune_epochs < 1 or args.patience < 1
            or args.batch_size < 1 or args.hidden_size < 1 or args.unfreeze_last_layers < 1
            or args.num_workers < 0 or args.benchmark_batches < 0):
        raise ValueError("Epochs, patience, batch size, hidden size and unfreeze count must be positive")
    if not 0 < args.backbone_learning_rate < args.head_learning_rate or args.weight_decay < 0:
        raise ValueError("Learning rates must be positive, with backbone rate smaller than head rate")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if args.benchmark_batches and len(args.seeds) != 1:
        raise ValueError("Benchmark mode requires exactly one seed")
    args.output_dir = (args.output_dir if args.output_dir is not None
                       else args.root / "data" / "processed" / "models" / "finetune_framing")
    splits = {name: load_split(args.root, name) for name in ("train", "val", "test")}
    baseline = baseline_metrics(args.baseline_matrix, splits["test"]) if args.baseline_matrix else None
    device = select_device()
    print(f"Device: {device}; model: {args.model_name}", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    results = [run_seed(copy.copy(args), seed, splits, processor, device) for seed in args.seeds]
    if not args.benchmark_batches:
        summary = {
            "seeds": args.seeds,
            "test_accuracy_mean": float(np.mean([r["test_strict_accuracy"] for r in results])),
            "test_accuracy_std": float(np.std([r["test_strict_accuracy"] for r in results])),
            "test_macro_f1_mean": float(np.mean([r["test_strict_macro_f1"] for r in results])),
            "test_macro_f1_std": float(np.std([r["test_strict_macro_f1"] for r in results])),
        }
        if baseline is not None:
            summary["baseline"] = baseline
            summary["baseline_note"] = "Matrix support matches test labels; individual examples were not verified."
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
