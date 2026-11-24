import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from movieshot.models import (
    FramingClassifier,
    SigLIPShotSizeClassifier,
    load_framing_model,
    load_shot_size_models,
    model_probabilities,
)
from scripts.paths import (
    embedding_file, ensemble_artifacts, framing_run, report_path, resolve_image_path,
    shot_size_runs, split_file,
)
from scripts import paths
from movieshot.training import shotsize
from movieshot.evaluation import compare_shotsize_models
from movieshot.evaluation import ensemble_seeds


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=3))

    def get_image_features(self, pixel_values):
        return pixel_values


class FakeProcessor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.ones((len(images), 3))}


class PathsAndModelsTests(unittest.TestCase):
    def test_raw_dataset_directory_is_preferred_over_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "data" / "raw"
            (root / "ShotBench").mkdir()
            (raw / "ShotBench").mkdir(parents=True)
            with patch.object(paths, "ROOT", root), patch.object(paths, "DATA_RAW_DIR", raw):
                self.assertEqual(paths._input_dir("ShotBench"), raw / "ShotBench")

    def test_relocated_images_preserve_old_paths_and_ignore_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            qa = root / "data" / "raw" / "data_shotqa" / "frame.jpg"
            bench = root / "data" / "raw" / "ShotBench" / "image" / "test.jpg"
            for path in (qa, bench):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(resolve_image_path("data_shotqa/frame.jpg", root), qa)
            self.assertEqual(resolve_image_path("/old/project/data_shotqa/frame.jpg", root), qa)
            self.assertEqual(resolve_image_path(r"C:\old\project\ShotBench\image\test.jpg", root), bench)
            self.assertEqual(resolve_image_path("data/raw/ShotBench/image/test.jpg", root), bench)
            unrelated = "/other/data_shotqa/frame.jpg"
            self.assertNotEqual(resolve_image_path(unrelated, root), qa)
            self.assertFalse(resolve_image_path("data_shotqa/missing.jpg", root).is_file())

    def test_move_preserves_comparison_cache_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / "data_shotqa" / "frame.jpg"
            legacy.parent.mkdir()
            legacy.write_bytes(b"image")
            frame = pd.DataFrame({"path": [str(legacy)]})
            with patch.object(compare_shotsize_models, "resolve_image_path",
                              side_effect=lambda path: resolve_image_path(path, root)):
                before = compare_shotsize_models.frame_fingerprint(frame)
                moved = root / "data" / "raw" / "data_shotqa" / "frame.jpg"
                moved.parent.mkdir(parents=True)
                legacy.rename(moved)
                self.assertEqual(compare_shotsize_models.frame_fingerprint(frame), before)

    def test_canonical_paths_preferred_and_legacy_files_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind, resolver, filename in (
                ("splits", split_file, "train_shot_size.csv"),
                ("emb", embedding_file, "train_shot_size.npy"),
            ):
                legacy = root / kind / filename
                legacy.parent.mkdir(parents=True)
                legacy.touch()
                self.assertEqual(resolver(filename, root), legacy)
                canonical = root / "data" / "processed" / kind / filename
                canonical.parent.mkdir(parents=True)
                canonical.touch()
                self.assertEqual(resolver(filename, root), canonical)

    def test_checkpoint_discovery_prefers_new_location_then_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / "data" / "processed" / "finetune_shotsize_protocol" / "seed1"
            legacy.mkdir(parents=True)
            (legacy / "best_model.pt").touch()
            self.assertEqual(shot_size_runs(root), [legacy])
            single = root / "outputs" / "finetune_shotsize"
            single.mkdir(parents=True)
            (single / "best_model.pt").touch()
            self.assertEqual(shot_size_runs(root), [single])
            ensemble = root / "outputs" / "finetune_shotsize_protocol" / "seed2"
            ensemble.mkdir(parents=True)
            (ensemble / "best_model.pt").touch()
            self.assertEqual(shot_size_runs(root), [ensemble])
            canonical = root / "data" / "processed" / "models" / "finetune_shotsize"
            for name in ("seed2", "seed1"):
                seed = canonical / name
                seed.mkdir(parents=True)
                (seed / "best_model.pt").touch()
            self.assertEqual(shot_size_runs(root), [canonical / "seed1", canonical / "seed2"])
            (canonical / "best_model.pt").touch()
            self.assertEqual(shot_size_runs(root), [canonical])

    def test_framing_checkpoint_prefers_canonical_and_supports_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "data" / "processed" / "models" / "finetune_framing" / "seed_42"
            legacy = root / "outputs" / "finetune_framing" / "seed_42"
            self.assertEqual(framing_run(root), canonical)
            for directory in (legacy, canonical):
                directory.mkdir(parents=True)
                (directory / "best_finetune.pt").touch()
                self.assertEqual(framing_run(root), directory)

    def test_report_routing_preserves_experiment_and_custom_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "outputs"
            self.assertEqual(report_path(output, "x.csv", root), output / "csv" / "x.csv")
            self.assertEqual(report_path(output / "experiment", "x.json", root),
                             output / "json" / "experiment" / "x.json")
            self.assertEqual(report_path(output / "csv" / "experiment", "x.json", root),
                             output / "json" / "experiment" / "x.json")
            custom = root / "custom"
            self.assertEqual(report_path(custom, "x.json", root), custom / "x.json")

    def test_ensemble_reports_and_legacy_lookup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            run.mkdir()
            (run / "class_mapping.json").write_text('{"a": 0, "b": 1}', encoding="utf-8")
            for split in ("val", "test"):
                pd.DataFrame({"path": ["x", "y"], "label": ["a", "b"]}).to_csv(
                    run / f"predictions_{split}.csv", index=False)
                np.save(run / f"logits_{split}.npy", np.array([[2., 0.], [0., 2.]]))
            legacy = root / "outputs" / "shotsize_ensemble" / "ensemble"
            legacy.mkdir(parents=True)
            (legacy / "ensemble_metrics.json").write_text("{}", encoding="utf-8")
            self.assertEqual(ensemble_artifacts(root), (legacy / "ensemble_metrics.json", legacy, legacy))
            output = root / "outputs" / "shotsize_ensemble"
            with patch.object(ensemble_seeds, "SHOTSIZE_ENSEMBLE_DIR", output), \
                 patch.object(ensemble_seeds, "DATA_PROCESSED_DIR", root / "data" / "processed"), \
                 patch.object(ensemble_seeds, "prepare_report",
                              side_effect=lambda directory, name: paths.prepare_report(directory, name, root)), \
                 patch.object(sys, "argv", ["ensemble", "--runs", str(run)]):
                ensemble_seeds.main()
            metrics, predictions, probabilities = ensemble_artifacts(root)
            self.assertEqual(metrics, root / "outputs" / "json" / "shotsize_ensemble" / "ensemble" / "ensemble_metrics.json")
            self.assertTrue(metrics.is_file())
            self.assertTrue((predictions / "predictions_val_plain.csv").is_file())
            self.assertTrue((probabilities / "probs_val_plain.npy").is_file())

    def test_checkpoint_loaders_and_probabilities(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = {"a": 0, "b": 1}
            shot = SigLIPShotSizeClassifier(FakeBackbone(), 3, 2)
            framing = FramingClassifier(FakeBackbone(), 3, 4, 2)
            torch.save({"model_state": shot.state_dict(), "label_to_id": labels, "model_name": "fake"},
                       root / "best_model.pt")
            torch.save({"model_state": framing.state_dict(), "label_to_id": labels, "model_name": "fake",
                        "hidden_size": 4}, root / "framing.pt")
            with patch("movieshot.models.AutoModel.from_pretrained", side_effect=lambda *a: FakeBackbone()), \
                 patch("movieshot.models.AutoImageProcessor.from_pretrained", return_value=FakeProcessor()):
                models, processor, loaded_labels = load_shot_size_models([root], torch.device("cpu"))
                loaded_framing, _, classes = load_framing_model(root / "framing.pt", torch.device("cpu"))
            self.assertEqual(loaded_labels, labels)
            self.assertEqual(classes, ["a", "b"])
            images = [Image.new("RGB", (4, 4))]
            for model in (models[0], loaded_framing):
                result = model_probabilities([model], processor, torch.device("cpu"), images)
                self.assertEqual(result.shape, (1, 1, 2))
                np.testing.assert_allclose(result.sum(axis=2), 1.0, atol=1e-6)

    def test_training_checkpoint_can_be_loaded_for_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "frame.png"
            Image.new("RGB", (4, 4)).save(image)
            frame = pd.DataFrame({"path": [str(image), str(image)], "label": ["a", "b"]})
            argv = ["train", "--epochs", "1", "--batch-size", "2",
                    "--model-name", "fake", "--output-dir", str(root)]
            with patch.object(sys, "argv", argv), \
                 patch.object(shotsize, "load_split", return_value=frame), \
                 patch("movieshot.models.AutoModel.from_pretrained", side_effect=lambda *a, **k: FakeBackbone()), \
                 patch("movieshot.models.AutoImageProcessor.from_pretrained", return_value=FakeProcessor()):
                self.assertEqual(shotsize.main(), 0)
                models, processor, labels = load_shot_size_models([root], torch.device("cpu"))
            self.assertEqual(labels, {"a": 0, "b": 1})
            with Image.open(image) as opened:
                result = model_probabilities(models, processor, torch.device("cpu"), [opened])
            self.assertEqual(result.shape, (1, 1, 2))


if __name__ == "__main__":
    unittest.main()
