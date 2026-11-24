import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from scripts.demo import app as demo


class FakeModel(nn.Module):
    def forward(self, inputs):
        return torch.tensor([[2.0, 1.0]]).repeat(len(inputs["pixel_values"]), 1)


class FakeProcessor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.zeros((len(images), 3))}


class DemoLoadingTests(unittest.TestCase):
    def test_framing_borderline_note_applies_to_any_close_pair(self):
        image = Image.new("RGB", (4, 4))
        feat = {"classes": ["single", "insert", "group shot"],
                "order": ["single", "insert", "group shot"]}
        with patch.object(demo, "model_probs") as probabilities, \
             patch.object(demo, "occlusion", return_value=[]), \
             patch.object(demo, "neighbours", return_value=[]), \
             patch.object(demo, "reliability", return_value=None):
            probabilities.return_value = np.array([[[0.45, 0.30, 0.25]]])
            result = demo.classify("framing", feat, image)
            self.assertEqual(result["note"],
                             "Borderline between “single” and “insert”, which are ambiguous in this picture.")
            probabilities.return_value = np.array([[[0.70, 0.20, 0.10]]])
            self.assertIsNone(demo.classify("framing", feat, image)["note"])

    def test_framing_borderline_preserves_known_confusion_warning(self):
        feat = {"classes": ["3 shot", "group shot"], "order": ["3 shot", "group shot"]}
        with patch.object(demo, "model_probs", return_value=np.array([[[0.55, 0.45]]])), \
             patch.object(demo, "occlusion", return_value=[]), \
             patch.object(demo, "neighbours", return_value=[]), \
             patch.object(demo, "reliability", return_value=None):
            note = demo.classify("framing", feat, Image.new("RGB", (4, 4)))["note"]
        self.assertIn("ambiguous in this picture", note)
        self.assertIn("model often confuses", note)

    def test_neighbour_index_resolves_paths_from_before_move(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            splits = root / "data" / "processed" / "splits"
            embeddings = root / "data" / "processed" / "emb"
            image = root / "data" / "raw" / "data_shotqa" / "frame.png"
            for directory in (splits, embeddings, image.parent):
                directory.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(image)
            for split in ("train", "val"):
                pd.DataFrame({"path": ["/old/project/data_shotqa/frame.png"], "label": ["close up"]}).to_csv(
                    splits / f"{split}_shot_size.csv", index=False
                )
                np.save(embeddings / f"{split}_shot_size.npy", np.ones((1, 3)))
            with patch.object(demo, "ROOT", root), patch.object(demo, "SPLITS_DIR", splits), \
                 patch.object(demo, "DATA_PROCESSED_DIR", root / "data" / "processed"):
                index = demo.load_index("shot_size")
            self.assertEqual(index["paths"], [str(image), str(image)])
            demo.STATE["shot_size"] = {"index": index}
            try:
                self.assertEqual(Path(demo.thumb("shot_size", 0).path), image)
            finally:
                demo.STATE.clear()

    def test_discovers_model_and_analyzes_uploaded_image(self):
        demo.STATE.clear()
        model = FakeModel()
        processor = FakeProcessor()
        with patch.object(demo, "select_device", return_value=torch.device("cpu")), \
             patch.object(demo, "shot_size_runs", return_value=[Path("legacy/seed1")]), \
             patch.object(demo, "load_shot_size_models",
                          return_value=([model], processor, {"close up": 0, "medium": 1})) as loader, \
             patch.object(demo, "load_framing_model", side_effect=FileNotFoundError("no framing checkpoint")), \
             patch.object(demo, "load_index", return_value=None), \
             patch.object(demo, "look", return_value={"lighting": "test"}):
            with TestClient(demo.app) as client:
                self.assertEqual(client.get("/status").json(), {"features": ["shot_size"]})
                image = io.BytesIO()
                Image.new("RGB", (32, 32)).save(image, format="PNG")
                response = client.post("/analyze", files={"file": ("image.png", image.getvalue(), "image/png")})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["shot_size"]["label"], "close up")
                self.assertEqual(len(response.json()["shot_size"]["heat"]), 8)
                self.assertIsNone(response.json()["framing"])
            loader.assert_called_once_with([Path("legacy/seed1")], torch.device("cpu"))
        demo.STATE.clear()


if __name__ == "__main__":
    unittest.main()
