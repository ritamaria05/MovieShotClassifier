"""Shared model definitions and checkpoint loading for inference."""

from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel

from scripts.paths import framing_run, shot_size_runs

MODEL_REPO = "hugomelo17/MovieShotClassifier"
MODEL_REVISION = "4c9571c3492102ff519bcb12a976f624c6bcd7ea"

SHOT_SIZE_ORDER = [
    "extreme close up",
    "close up",
    "medium close up",
    "medium",
    "medium wide",
    "wide",
    "extreme wide",
]


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


class FramingClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_classes)
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.backbone.get_image_features(**inputs)
        if not torch.is_tensor(features):
            features = features.pooler_output
        return self.classifier(nn.functional.normalize(features.float(), dim=-1))


def shot_size_checkpoints(single: bool = False) -> list[Path]:
    runs = shot_size_runs()
    if runs:
        return [run / "best_model.pt" for run in (runs[:1] if single else runs)]
    print(f"No local shot-size checkpoint found; downloading from {MODEL_REPO}.")
    seeds = range(1, 2 if single else 6)
    return [Path(hf_hub_download(MODEL_REPO, f"shotsize_model_{seed}.pt", revision=MODEL_REVISION))
            for seed in seeds]


def framing_checkpoint() -> Path:
    local = framing_run() / "best_finetune.pt"
    if local.is_file():
        return local
    print(f"No local framing checkpoint found; downloading from {MODEL_REPO}.")
    return Path(hf_hub_download(MODEL_REPO, "framing_model.pt", revision=MODEL_REVISION))


def load_shot_size_models(run_dirs: list[Path], device: torch.device):
    return load_shot_size_checkpoints([run / "best_model.pt" for run in run_dirs], device)


def load_shot_size_checkpoints(checkpoint_paths: list[Path], device: torch.device):
    if not checkpoint_paths:
        raise ValueError("No shot-size checkpoints provided")
    models, label_to_id, model_name = [], None, None
    for path in checkpoint_paths:
        print(f"Loading {path} ...")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if label_to_id is None:
            label_to_id, model_name = checkpoint["label_to_id"], checkpoint["model_name"]
        elif checkpoint["label_to_id"] != label_to_id or checkpoint["model_name"] != model_name:
            raise ValueError(f"{path}: checkpoint classes or model name differ from the other runs")
        backbone = AutoModel.from_pretrained(model_name)
        feature_dim = getattr(backbone.config, "projection_dim", None)
        if feature_dim is None:
            feature_dim = backbone.config.vision_config.hidden_size
        model = SigLIPShotSizeClassifier(backbone, feature_dim, len(label_to_id))
        model.load_state_dict(checkpoint["model_state"])
        if hasattr(model.backbone, "text_model"):
            del model.backbone.text_model
        models.append(model.to(device).eval())
    return models, AutoImageProcessor.from_pretrained(model_name), label_to_id


def load_framing_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    backbone = AutoModel.from_pretrained(checkpoint["model_name"])
    model = FramingClassifier(backbone, backbone.config.vision_config.hidden_size,
                              checkpoint["hidden_size"], len(checkpoint["label_to_id"]))
    model.load_state_dict(checkpoint["model_state"])
    if hasattr(model.backbone, "text_model"):
        del model.backbone.text_model
    classes = [c for c, _ in sorted(checkpoint["label_to_id"].items(), key=lambda item: item[1])]
    processor = AutoImageProcessor.from_pretrained(checkpoint["model_name"])
    return model.to(device).eval(), processor, classes


@torch.no_grad()
def model_probabilities(models, processor, device: torch.device, images: list[Image.Image]) -> np.ndarray:
    inputs = {key: value.to(device) for key, value in processor(images=images, return_tensors="pt").items()}
    outputs = []
    for model in models:
        logits = model(inputs)
        logits = getattr(logits, "logits", logits)
        outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.stack(outputs)
