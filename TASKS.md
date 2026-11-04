# MovieShotClassifier task checklist

## Completed

- [x] Define the project goal: classify movie frames by shot size and framing.
- [x] Add scripts for parsing ShotQA SFT conversations.
- [x] Combine SFT and GRPO labels for shot size and framing.
- [x] Download and save ShotBench annotations.
- [x] Add metadata and overlap diagnostics.
- [x] Add film-level leakage prevention.
- [x] Create train/validation/test split generation.
- [x] Add SigLIP image embedding extraction.
- [x] Cache shot-size embeddings in `emb/`.
- [x] Add a shot-size logistic-regression evaluation baseline.
- [x] Document the pipeline in `PROJECT_GUIDE.md`.

## Required setup on a normal PC

- [ ] Install Python 3.10 or newer.
- [ ] Create a virtual environment.
- [ ] Install the project dependencies, including `scikit-learn`, `torch`,
  and `transformers`.
- [ ] Obtain the ShotQA metadata files:
  `ShotQA_meta/meta.jsonl`, `sft.json`, and `grpo.json`.
- [ ] Obtain or extract the ShotQA images into `data_shotqa/`.
- [ ] Obtain the ShotBench images in `ShotBench/image/`.
- [ ] Record the Python version, package versions, dataset versions, and model
  revision so the experiment can be reproduced.

## Required data-pipeline tasks

- [ ] Run `make_annotations.py`.
- [ ] Run `build_labels.py`.
- [ ] Run `final_labels.py`.
- [ ] Download/extract every image listed in `lista_extrair.txt`.
- [ ] Run `verify.py` and resolve missing images.
- [ ] Run `check_meta.py` and inspect label/metadata disagreements.
- [ ] Run `check_overlap.py` and investigate unexpected overlap.
- [ ] Run `build_dataset.py`.
- [ ] Confirm that the generated split files contain the expected classes and
  that films do not cross from training into the ShotBench test set.

## Baseline evaluation tasks

- [ ] Run `embed_shotsize.py`.
- [ ] Record accuracy, macro-F1, confusion matrix, and near-duplicate counts.
- [ ] Confirm that the `.npy` embedding row order matches the split CSV order.
- [ ] Add a majority-class baseline for comparison.
- [ ] Repeat the evaluation with the same seed and verify that results are
  consistent.

## Tasks needed for a complete project

- [ ] Evaluate `framing`, not only `shot_size`.
- [ ] Decide whether `camera_angle` is part of the project scope.
- [ ] Save predictions and metrics to files instead of printing them only.
- [ ] Save the trained logistic-regression classifier.
- [ ] Add tests for label parsing, category filtering, split leakage, missing
  files, and multi-label examples.
- [ ] Add all direct dependencies and exact versions to reproducible dependency
  files.
- [ ] Document the dataset sources, expected counts, and checksums.
- [ ] Remove the tracked `.DS_Store`.
- [ ] Update `README.md` with setup, commands, results, and limitations.

## Optional deep-learning tasks

These are not required to make the current baseline work:

- [ ] Train a small neural classification head on frozen SigLIP embeddings.
- [ ] Fine-tune only the last SigLIP layers.
- [ ] Compare frozen features, partial fine-tuning, and full fine-tuning.
- [ ] Use early stopping, checkpointing, and validation monitoring.
- [ ] Use a multi task learning model to predict shot size and framing simultaneously.
- [ ] Compare all models on the same film-separated test set.

## Optional tasks
- [ ] Add other characteristics like color palette, brightness, or motion blur, which are easier to implement, and show other aspects of computer vision.

## What is realistic on a normal PC?

The data parsing, validation, split generation, and logistic-regression
baseline are realistic on a normal PC. Embedding extraction is also possible,
but CPU execution may be slow and requires substantial disk space and RAM.

Fine-tuning SigLIP is substantially more demanding. A dedicated GPU with
sufficient VRAM is recommended. Without one, start with the frozen-embedding
baseline, reduce the dataset size for experiments, or use a smaller pretrained
vision model.

The minimum sensible completion path is:

1. Finish the local data setup.
2. Generate and verify the splits.
3. Run and record the shot-size baseline.
4. Evaluate framing.
5. Add reproducibility files and tests.
6. Only then decide whether fine-tuning is necessary.
