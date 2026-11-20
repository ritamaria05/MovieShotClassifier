import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

import MovieShotClassifier.scripts.framing.finetune_framing as framing


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
        self.config = argparse.Namespace(vision_config=argparse.Namespace(hidden_size=3))

    def get_image_features(self, pixel_values):
        for layer in self.vision_model.encoder.layers:
            pixel_values = torch.relu(layer(pixel_values))
        return pixel_values


class FakeProcessor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.tensor(np.array([np.asarray(i).mean(axis=(0, 1)) for i in images]),
                                             dtype=torch.float32) / 255}


class FramingTests(unittest.TestCase):
    def test_freezing_and_unfreezing(self):
        model = framing.FramingClassifier(FakeBackbone(), 3, 4, 2)
        framing.configure_trainable_layers(model, 0)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        framing.configure_trainable_layers(model, 1)
        layers = model.backbone.vision_model.encoder.layers
        self.assertTrue(all(not p.requires_grad for p in layers[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in layers[1].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        with self.assertRaises(ValueError):
            framing.configure_trainable_layers(model, 3)

    def test_staged_run_and_multilabel_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "splits").mkdir()
            (root / "images").mkdir()
            for name in ("train", "val", "test"):
                rows = []
                labels = ["a", "b", "a, b"] if name == "test" else ["a", "b", "a", "b"]
                for i, label in enumerate(labels):
                    path = f"images/{name}{i}.png"
                    Image.new("RGB", (4, 4), (i * 50, 20, 50)).save(root / path)
                    rows.append({"path": path, "label": label})
                pd.DataFrame(rows).to_csv(root / "splits" / f"{name}_framing.csv", index=False)
            splits = {name: framing.load_split(root, name) for name in ("train", "val", "test")}
            args = argparse.Namespace(root=root, output_dir=root / "results", model_name="fake",
                hidden_size=4, batch_size=2, num_workers=0, benchmark_batches=0,
                head_epochs=2, finetune_epochs=2, patience=1, unfreeze_last_layers=1,
                head_learning_rate=0.01, backbone_learning_rate=0.001, weight_decay=0)
            with patch.object(framing.AutoModel, "from_pretrained", side_effect=lambda *a: FakeBackbone()):
                result = framing.run_seed(args, 42, splits, FakeProcessor(), torch.device("cpu"))
            output = root / "results" / "seed_42"
            for name in ("best_head.pt", "best_finetune.pt", "confusion_matrix.csv",
                         "predictions_test.csv", "metrics.json"):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(result["test_n_strict"], 2)
            self.assertEqual(result["test_lenient_accuracy"], 2 / 3)
            self.assertEqual(len(pd.read_csv(output / "predictions_test.csv")), 3)
            checkpoint = torch.load(output / "best_finetune.pt", weights_only=False)
            self.assertEqual(checkpoint["label_to_id"], {"a": 0, "b": 1})
            saved = json.loads((output / "metrics.json").read_text())
            self.assertEqual(saved["finetune"]["best_epoch"], 1)
            baseline = framing.baseline_metrics(output / "confusion_matrix.csv", splits["test"])
            self.assertEqual(baseline["accuracy"], result["test_strict_accuracy"])
            self.assertEqual(baseline["macro_f1"], result["test_strict_macro_f1"])


if __name__ == "__main__":
    unittest.main()
