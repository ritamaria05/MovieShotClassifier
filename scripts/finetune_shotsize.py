"""Fine-tune SigLIP for shot-size classification.

The default mode trains only a linear classification head on frozen SigLIP
image features. ``--mode last_layers`` additionally unfreezes the final
vision encoder layers.

Optional training-recipe flags (all off by default, so the default behaviour
is the same as before):
  --head-lr              separate learning rate for the new linear head
  --scheduler cosine     cosine decay, with --warmup-ratio of linear warmup
  --lp-epochs N          linear probing first (backbone frozen) for N epochs,
                         then fine-tuning for --epochs epochs (LP-FT)
  --unfreeze-post-layers also train vision_model.post_layernorm and the
                         attention-pooling head (needs --mode last_layers)
  --debug-samples N      use N random rows per split (quick smoke test)

Besides predictions_test.csv, the script writes predictions_val.csv,
logits_val.npy, logits_test.npy and config.json to the output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="fine-tuning epochs (any --lp-epochs come before these)",
    )
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="learning rate of the new linear head (default: --learning-rate)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="fraction of fine-tuning steps used for linear warmup (cosine only)",
    )
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
    parser.add_argument(
        "--unfreeze-post-layers",
        action="store_true",
        help="also train vision_model.post_layernorm and the pooling head "
        "(requires --mode last_layers)",
    )
    parser.add_argument(
        "--lp-epochs",
        type=int,
        default=0,
        help="linear-probing epochs (backbone frozen, head only) run before "
        "fine-tuning; requires --mode last_layers",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=0,
        help="if > 0, use this many random rows per split (smoke test only)",
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
    missing_images = [p for p in frame["path"] if not Path(p).is_file()]
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
    model: SigLIPShotSizeClassifier,
    mode: str,
    layer_count: int,
    unfreeze_post: bool = False,
) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if mode == "head":
        return
    if layer_count < 1:
        raise ValueError("--unfreeze-last-layers must be at least 1 in last_layers mode")
    vision_model = getattr(model.backbone, "vision_model", None)
    layers = getattr(getattr(vision_model, "encoder", None), "layers", None)
    if layers is None or layer_count > len(layers):
        raise ValueError("Could not locate the requested SigLIP vision encoder layers")
    for layer in layers[-layer_count:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    projection = getattr(model.backbone, "visual_projection", None)
    if projection is not None:
        for parameter in projection.parameters():
            parameter.requires_grad = True
    if unfreeze_post:
        for name in ("post_layernorm", "head"):
            module = getattr(vision_model, name, None)
            if module is None:
                print(f"WARNING: vision_model.{name} not found; nothing to unfreeze for it")
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True


def build_optimizer(
    model: SigLIPShotSizeClassifier,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_params = list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    groups: list[dict[str, Any]] = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def cosine_warmup_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def make_scheduler(
    optimizer: torch.optim.Optimizer, kind: str, warmup_ratio: float, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if kind == "none":
        return None
    warmup_steps = int(round(warmup_ratio * total_steps))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_warmup_factor(step, warmup_steps, total_steps)
    )


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def run_epoch(
    model: SigLIPShotSizeClassifier,
    loader: DataLoader,
    processor: Any,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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
                if scheduler is not None:
                    scheduler.step()
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
    flip: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the frame with a `prediction` column and the raw logits."""
    dataset = ShotSizeDataset(frame, label_to_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for images, _, _ in tqdm(loader, desc="predict", leave=False):
            images = list(images)
            if flip:
                images = [ImageOps.mirror(image) for image in images]
            inputs = move_inputs(processor(images=images, return_tensors="pt"), device)
            chunks.append(model(inputs).float().cpu().numpy())
    logits = np.concatenate(chunks, axis=0)
    classes = list(label_to_id)
    result = frame.copy()
    result["prediction"] = [classes[index] for index in logits.argmax(axis=1)]
    return result, logits


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.patience < 1:
        raise ValueError("--patience must be positive")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative")
    if args.lp_epochs < 0:
        raise ValueError("--lp-epochs must be non-negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.head_lr is not None and args.head_lr <= 0:
        raise ValueError("--head-lr must be positive")
    if args.lp_epochs > 0 and args.mode != "last_layers":
        raise ValueError("--lp-epochs requires --mode last_layers")
    if args.unfreeze_post_layers and args.mode != "last_layers":
        raise ValueError("--unfreeze-post-layers requires --mode last_layers")
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
    if args.debug_samples > 0:
        print(f"DEBUG: using {args.debug_samples} random rows per split")
        splits = {
            name: frame.sample(
                n=min(args.debug_samples, len(frame)), random_state=args.seed
            ).reset_index(drop=True)
            for name, frame in splits.items()
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "class_mapping.json").write_text(
        json.dumps(label_to_id, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    device = select_device()
    print(f"Loading {args.model_name} on {device} ({args.mode} mode)")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)
    feature_dim = getattr(backbone.config, "projection_dim", None)
    if feature_dim is None:
        feature_dim = backbone.config.vision_config.hidden_size
    model = SigLIPShotSizeClassifier(backbone, feature_dim, len(label_to_id)).to(device)

    train_loader = DataLoader(
        ShotSizeDataset(splits["train"], label_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    counts = splits["train"]["label"].value_counts()
    weights = torch.tensor(
        [len(splits["train"]) / (len(label_to_id) * counts.get(label, 1)) for label in label_to_id],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    head_lr = args.head_lr if args.head_lr is not None else args.learning_rate

    plan: list[tuple[str, int]] = []
    if args.lp_epochs > 0:
        plan.append(("lp", args.lp_epochs))
    plan.append(("ft", args.epochs))
    total_epochs = sum(n for _, n in plan)

    history: list[dict[str, float | int | str]] = []
    best_f1 = -float("inf")
    best_epoch = 0
    best_path = args.output_dir / "best_model.pt"
    epochs_without_improvement = 0
    epoch = 0
    stopped = False
    for stage, stage_epochs in plan:
        if stage == "lp":
            configure_trainable_layers(model, "head", args.unfreeze_last_layers)
            optimizer = build_optimizer(model, head_lr, head_lr, args.weight_decay)
            scheduler = None
            print(f"Stage lp: linear probing for {stage_epochs} epoch(s), head lr={head_lr:g}")
        else:
            configure_trainable_layers(
                model, args.mode, args.unfreeze_last_layers, args.unfreeze_post_layers
            )
            optimizer = build_optimizer(model, args.learning_rate, head_lr, args.weight_decay)
            scheduler = make_scheduler(
                optimizer,
                args.scheduler,
                args.warmup_ratio,
                len(train_loader) * stage_epochs,
            )
            print(
                f"Stage ft: {stage_epochs} epoch(s), backbone lr={args.learning_rate:g}, "
                f"head lr={head_lr:g}, scheduler={args.scheduler}"
            )
        for _ in range(stage_epochs):
            epoch += 1
            train_loss, train_f1 = run_epoch(
                model, train_loader, processor, device, criterion, optimizer, scheduler
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
                "stage": stage,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
            history.append(row)
            print(
                f"epoch {epoch}/{total_epochs} [{stage}]: train loss={train_loss:.4f}, "
                f"train F1={train_f1:.4f}, val loss={val_loss:.4f}, val F1={val_f1:.4f}"
            )
            if val_f1 > best_f1 + args.min_delta:
                best_f1 = val_f1
                best_epoch = epoch
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
                        "stage": stage,
                        "val_macro_f1": val_f1,
                    },
                    best_path,
                )
            elif stage == "ft":
                epochs_without_improvement += 1
                print(
                    f"No validation improvement for "
                    f"{epochs_without_improvement}/{args.patience} epoch(s)"
                )
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    stopped = True
                    break
        if stopped:
            break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_frame = splits["test"]
    single = ~test_frame["label"].str.contains(",")
    test_predictions, test_logits = predict(
        model, test_frame, label_to_id, processor, device, args.batch_size, args.num_workers
    )
    test_predictions.to_csv(args.output_dir / "predictions_test.csv", index=False)
    np.save(args.output_dir / "logits_test.npy", test_logits)
    val_predictions, val_logits = predict(
        model, splits["val"], label_to_id, processor, device, args.batch_size, args.num_workers
    )
    val_predictions.to_csv(args.output_dir / "predictions_val.csv", index=False)
    np.save(args.output_dir / "logits_val.npy", val_logits)
    evaluated = test_predictions.loc[single]
    test_metrics = {
        "accuracy": float(accuracy_score(evaluated["label"], evaluated["prediction"])),
        "macro_f1": float(f1_score(evaluated["label"], evaluated["prediction"], average="macro")),
        "evaluated_single_label_examples": int(len(evaluated)),
        "excluded_multi_label_examples": int((~single).sum()),
        "best_val_macro_f1": float(best_f1),
        "best_epoch": int(best_epoch),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(test_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())