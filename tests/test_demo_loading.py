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
    def test_demo_uses_hugging_face_even_when_local_models_exist(self):
        demo.STATE.clear()
        model, processor = FakeModel(), FakeProcessor()
        with tempfile.TemporaryDirectory() as temp:
            local = Path(temp) / "local"
            local.mkdir()
            (local / "best_model.pt").touch()
            (local / "best_finetune.pt").touch()
            with patch("movieshot.models.shot_size_runs", return_value=[local]) as shot_runs, \
                 patch("movieshot.models.framing_run", return_value=local) as framing_run, \
                 patch("movieshot.models.hf_hub_download",
                       side_effect=lambda repo, name, revision: str(Path(temp) / name)) as download, \
                 patch.object(demo, "select_device", return_value=torch.device("cpu")), \
                 patch.object(demo, "load_shot_size_checkpoints",
                              return_value=([model], processor, {"close up": 0, "medium": 1})) as shot_loader, \
                 patch.object(demo, "load_framing_model",
                              return_value=(model, processor, ["single", "insert"])) as frame_loader, \
                 patch.object(demo, "load_index", return_value=None), \
                 patch.object(demo, "look", return_value=None):
                with TestClient(demo.app) as client:
                    self.assertEqual(client.get("/status").json(), {"features": ["shot_size", "framing"]})
                    image = io.BytesIO()
                    Image.new("RGB", (32, 32)).save(image, format="PNG")
                    response = client.post("/analyze", files={"file": ("image.png", image.getvalue(), "image/png")})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["shot_size"]["label"], "close up")
                    self.assertEqual(response.json()["framing"]["label"], "single")
                    self.assertIsNone(response.json()["framing"]["reliability"])
                self.assertEqual(download.call_count, 6)
                self.assertEqual(shot_loader.call_args.args[0],
                                 [Path(temp) / f"shotsize_model_{seed}.pt" for seed in range(1, 6)])
                self.assertEqual(frame_loader.call_args.args[0], Path(temp) / "framing_model.pt")
                shot_runs.assert_not_called()
                framing_run.assert_not_called()
        demo.STATE.clear()

    def test_demo_does_not_fall_back_to_local_models_on_download_failure(self):
        with patch.dict(demo.STATE, {}, clear=True), \
             patch("movieshot.models.shot_size_runs") as shot_runs, \
             patch("movieshot.models.framing_run") as framing_run, \
             patch("movieshot.models.hf_hub_download", side_effect=OSError("offline")), \
             patch.object(demo, "select_device", return_value=torch.device("cpu")):
            with TestClient(demo.app) as client:
                self.assertEqual(client.get("/status").json(), {"features": []})
            shot_runs.assert_not_called()
            framing_run.assert_not_called()

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

    def test_thumbnail_uses_dataset_even_when_local_image_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            splits = root / "data" / "processed" / "splits"
            embeddings = root / "data" / "processed" / "emb"
            image = root / "data" / "raw" / "data_shotqa" / "FRAME.png"
            downloaded = root / "downloaded.png"
            for directory in (splits, embeddings, image.parent):
                directory.mkdir(parents=True)
            Image.new("RGB", (4, 4), "red").save(image)
            Image.new("RGB", (4, 4), "blue").save(downloaded)
            for split in ("train", "val"):
                pd.DataFrame({"path": [str(image)], "label": ["close up"]}).to_csv(
                    splits / f"{split}_shot_size.csv", index=False
                )
                np.save(embeddings / f"{split}_shot_size.npy", np.ones((1, 3)))
            with patch.object(demo, "ROOT", root), patch.object(demo, "SPLITS_DIR", splits), \
                 patch.object(demo, "DATA_PROCESSED_DIR", root / "data" / "processed"):
                index = demo.load_index("shot_size")
            self.assertEqual(index["remote_paths"], ["images/0-H/FRAME.png"] * 2)
            with patch.dict(demo.STATE, {"shot_size": {"index": index}}, clear=True), \
                 patch.object(demo, "hf_hub_download", return_value=str(downloaded)) as download:
                response = TestClient(demo.app).get("/thumb/shot_size/0")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, downloaded.read_bytes())
            download.assert_called_once_with(
                demo.DATASET_REPO, "images/0-H/FRAME.png", repo_type="dataset", revision=demo.DATASET_REVISION
            )

    def test_reference_image_paths_match_uploaded_folders(self):
        examples = {
            "/old/MovieShotClassifier/data_shotqa/0FRAME.jpg": "images/0-H/0FRAME.jpg",
            r"C:\old\MovieShotClassifier\data_shotqa\HFRAME.png": "images/0-H/HFRAME.png",
            "data/raw/data_shotqa/IFRAME.jpeg": "images/I-Z/IFRAME.jpeg",
            "data_shotqa/ZFRAME.jpg": "images/I-Z/ZFRAME.jpg",
        }
        for path, expected in examples.items():
            with self.subTest(path=path):
                self.assertEqual(demo.reference_image_path(path), expected)
        for path in ("other/data_shotqa/AFRAME.jpg", "ShotBench/AFRAME.jpg",
                     "data_shotqa/../AFRAME.jpg", "data_shotqa/sub/AFRAME.jpg",
                     "data_shotqa/aframe.jpg", "data_shotqa/A%2F.jpg"):
            with self.subTest(path=path):
                self.assertIsNone(demo.reference_image_path(path))

    def test_index_keeps_remote_paths_aligned_with_train_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            splits = root / "data" / "processed" / "splits"
            embeddings = root / "data" / "processed" / "emb"
            splits.mkdir(parents=True)
            embeddings.mkdir(parents=True)
            for split, name in (("train", "AFRAME.jpg"), ("val", "IFRAME.png")):
                pd.DataFrame({"path": [f"/old/project/data_shotqa/{name}"], "label": [split]}).to_csv(
                    splits / f"{split}_framing.csv", index=False
                )
                np.save(embeddings / f"{split}_framing.npy", np.ones((1, 3)))
            with patch.object(demo, "ROOT", root), patch.object(demo, "SPLITS_DIR", splits), \
                 patch.object(demo, "DATA_PROCESSED_DIR", root / "data" / "processed"):
                index = demo.load_index("framing")
            self.assertEqual(index["labels"], ["train", "val"])
            self.assertEqual(index["remote_paths"], ["images/0-H/AFRAME.jpg", "images/I-Z/IFRAME.png"])

    def test_thumbnail_downloads_images_from_both_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            downloaded = Path(temp) / "downloaded.png"
            Image.new("RGB", (4, 4), "blue").save(downloaded)
            index = {
                "remote_paths": ["images/0-H/HFRAME.png", "images/I-Z/IFRAME.jpeg"],
            }
            with patch.dict(demo.STATE, {"shot_size": {"index": index}}, clear=True), \
                 patch.object(demo, "hf_hub_download", return_value=str(downloaded)) as download:
                client = TestClient(demo.app)
                for i, remote in ((0, "images/0-H/HFRAME.png"), (1, "images/I-Z/IFRAME.jpeg")):
                    response = client.get(f"/thumb/shot_size/{i}")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content, downloaded.read_bytes())
                    download.assert_called_with(
                        demo.DATASET_REPO, remote, repo_type="dataset", revision=demo.DATASET_REVISION
                    )

    def test_thumbnail_errors_are_explicit(self):
        index = {"remote_paths": [None, "images/0-H/A.jpg"]}
        with patch.dict(demo.STATE, {"framing": {"index": index}}, clear=True), \
             patch.object(demo, "hf_hub_download", side_effect=OSError("offline")) as download:
            client = TestClient(demo.app)
            for url in ("/thumb/framing/0", "/thumb/framing/-1", "/thumb/framing/2", "/thumb/other/0"):
                self.assertEqual(client.get(url).status_code, 404)
            download.assert_not_called()
            response = client.get("/thumb/framing/1")
            self.assertEqual(response.status_code, 503)
            self.assertIn("Could not retrieve", response.json()["detail"])
            download.assert_called_once()

    def test_analyzes_image_when_only_shot_size_model_is_available(self):
        demo.STATE.clear()
        model = FakeModel()
        processor = FakeProcessor()
        with patch.object(demo, "select_device", return_value=torch.device("cpu")), \
             patch.object(demo, "download_shot_size_checkpoints", return_value=[Path("shotsize_model_1.pt")]), \
             patch.object(demo, "load_shot_size_checkpoints",
                          return_value=([model], processor, {"close up": 0, "medium": 1})) as loader, \
             patch.object(demo, "download_framing_checkpoint", return_value=Path("missing.pt")), \
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
            loader.assert_called_once_with([Path("shotsize_model_1.pt")], torch.device("cpu"))
        demo.STATE.clear()


if __name__ == "__main__":
    unittest.main()
