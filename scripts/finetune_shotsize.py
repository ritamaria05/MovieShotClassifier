"""Fine-tune SigLIP for shot-size classification.

The default mode trains only a linear classification head on frozen SigLIP
image features. ``--mode last_layers`` additionally unfreezes the final
vision encoder layers.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from paths import DATA_PROCESSED_DIR, SPLITS_DIR


DEFAULT_MODEL = "google/siglip-base-patch16-224"
DEFAULT_OUTPUT = DATA_PROCESSED_DIR / "finetune_shotsize"
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("head", "last_layers"),
        default="head",
        help="head trains only the classifier; last_layers also unfreezes the "
        "final vision encoder layers",
    )
    parser.add_argument(
        "--unfreeze-last-layers",
        type=int,
        default=2,
        help="number of final vision encoder layers for --mode last_layers",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ShotSizeDataset(Dataset[tuple[Image.Image, int, str]]):
    def __init__(self, frame: pd.DataFrame, label_to_id: dict[str, int]) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        row = self.frame.iloc[index]
        try:
            image = Image.open(row["path"]).convert("RGB")
        except (FileNotFoundError, OSError) as error:
            raise RuntimeError(f"Could not read image {row['path']}: {error}") from error
        return image, self.label_to_id.get(row["label"], -1), str(row["path"])


class SigLIPShotSizeClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.backbone.get_image_features(**inputs)
        if not torch.is_tensor(features):
            features = features.pooler_output
        return self.classifier(features.float())


def load_split(name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{name}_shot_size.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} split at {path}. Run `python scripts/build_dataset.py` "
            "after preparing the local datasets."
        )
    frame = pd.read_csv(path)
    required = {"path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} is empty")
    missing_images = [path for path in frame["path"] if not Path(path).is_file()]
    if missing_images:
        preview = ", ".join(map(str, missing_images[:3]))
        raise FileNotFoundError(
            f"{path} references {len(missing_images)} missing image(s), including {preview}"
        )
    frame["label"] = frame["label"].astype(str).str.strip()
    return frame


def collate(
    batch: list[tuple[Image.Image, int, str]],
) -> tuple[list[Image.Image], torch.Tensor, list[str]]:
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(paths)


def configure_trainable_layers(
    model: SigLIPShotSizeClassifier, mode: str, layer_count: int
) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if mode == "head":
        return
    if layer_count < 1:
        raise ValueError("--unfreeze-last-layers must be at least 1 in last_layers mode")
    layers = getattr(
        getattr(getattr(model.backbone, "vision_model", None), "encoder", None),
        "layers",
        None,
    )
    if layers is None or layer_count > len(layers):
        raise ValueError("Could not locate the requested SigLIP vision encoder layers")
    for layer in layers[-layer_count:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    projection = getattr(model.backbone, "visual_projection", None)
    if projection is not None:
        for parameter in projection.parameters():
            parameter.requires_grad = True


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def run_epoch(
    model: SigLIPShotSizeClassifier,
    loader: DataLoader,
    processor: Any,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _ in tqdm(loader, leave=False):
            inputs = move_inputs(processor(images=list(images), return_tensors="pt"), device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
    return float(np.mean(losses)), float(f1_score(targets, predictions, average="macro"))


def predict(
    model: SigLIPShotSizeClassifier,
    frame: pd.DataFrame,
    label_to_id: dict[str, int],
    processor: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    dataset = ShotSizeDataset(frame, label_to_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
    model.eval()
    predicted: list[int] = []
    with torch.no_grad():
        for images, _, _ in tqdm(loader, desc="predict", leave=False):
            inputs = move_inputs(processor(images=list(images), return_tensors="pt"), device)
            predicted.extend(model(inputs).argmax(dim=1).cpu().tolist())
    result = frame.copy()
    result["prediction"] = [list(label_to_id)[index] for index in predicted]
    return result


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.patience < 1:
        raise ValueError("--patience must be positive")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative")
    set_seed(args.seed)
    splits = {name: load_split(name) for name in SPLIT_NAMES}
    label_to_id = {label: index for index, label in enumerate(sorted(splits["train"]["label"].unique()))}
    unknown = {
        label
        for name in ("val", "test")
        for label in splits[name]["label"].unique()
        if "," not in label and label not in label_to_id
    }
    if unknown:
        raise ValueError(f"Validation/test contain labels absent from train: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "class_mapping.json").write_text(
        json.dumps(label_to_id, indent=2, sort_keys=True), encoding="utf-8"
    )

    device = select_device()
    print(f"Loading {args.model_name} on {device} ({args.mode} mode)")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)
    feature_dim = getattr(backbone.config, "projection_dim", None)
    if feature_dim is None:
        feature_dim = backbone.config.vision_config.hidden_size
    model = SigLIPShotSizeClassifier(backbone, feature_dim, len(label_to_id)).to(device)
    configure_trainable_layers(model, args.mode, args.unfreeze_last_layers)

    train_loader = DataLoader(
        ShotSizeDataset(splits["train"], label_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    counts = splits["train"]["label"].value_counts()
    weights = torch.tensor(
        [len(splits["train"]) / (len(label_to_id) * counts[label]) for label in label_to_id],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    best_f1 = -float("inf")
    best_path = args.output_dir / "best_model.pt"
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = run_epoch(
            model, train_loader, processor, device, criterion, optimizer
        )
        val_loader = DataLoader(
            ShotSizeDataset(splits["val"], label_to_id),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        val_loss, val_f1 = run_epoch(model, val_loader, processor, device, criterion, None)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs}: train loss={train_loss:.4f}, "
            f"train F1={train_f1:.4f}, val loss={val_loss:.4f}, val F1={val_f1:.4f}"
        )
        if val_f1 > best_f1 + args.min_delta:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "model_name": args.model_name,
                    "mode": args.mode,
                    "unfreeze_last_layers": args.unfreeze_last_layers,
                    "label_to_id": label_to_id,
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement for "
                f"{epochs_without_improvement}/{args.patience} epoch(s)"
            )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_frame = splits["test"]
    single = ~test_frame["label"].str.contains(",")
    test_predictions = predict(
        model, test_frame, label_to_id, processor, device, args.batch_size, args.num_workers
    )
    test_predictions.to_csv(args.output_dir / "predictions_test.csv", index=False)
    evaluated = test_predictions.loc[single]
    test_metrics = {
        "accuracy": float(accuracy_score(evaluated["label"], evaluated["prediction"])),
        "macro_f1": float(f1_score(evaluated["label"], evaluated["prediction"], average="macro")),
        "evaluated_single_label_examples": int(len(evaluated)),
        "excluded_multi_label_examples": int((~single).sum()),
        "best_val_macro_f1": float(best_f1),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(test_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
