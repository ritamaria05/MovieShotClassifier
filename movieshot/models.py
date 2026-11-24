"""Shared model definitions and checkpoint loading for inference."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel

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


def load_shot_size_models(run_dirs: list[Path], device: torch.device):
    if not run_dirs:
        raise ValueError("No shot-size checkpoint directories provided")
    models, label_to_id, model_name = [], None, None
    for run in run_dirs:
        print(f"Loading {run} ...")
        checkpoint = torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)
        if label_to_id is None:
            label_to_id, model_name = checkpoint["label_to_id"], checkpoint["model_name"]
        elif checkpoint["label_to_id"] != label_to_id or checkpoint["model_name"] != model_name:
            raise ValueError(f"{run}: checkpoint classes or model name differ from the other runs")
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
