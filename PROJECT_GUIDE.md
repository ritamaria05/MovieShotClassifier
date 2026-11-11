# MovieShotClassifier: Project Guide

## Theme and objective

This project studies **movie shot scale/size and framing** from individual movie
frames. The current pipeline combines ShotQA annotations with the ShotBench
test set, prevents movie-level leakage between splits, extracts SigLIP image
embeddings, and evaluates a logistic-regression baseline.

The repository currently contains the scripts and cached shot-size embeddings.
The source datasets, extracted frames, generated CSV files, and split files are
intentionally ignored by Git and must be obtained or generated locally.

## Required local data

Run the scripts from the repository root. The following paths are expected:

```text
ShotQA_meta/meta.jsonl        # or data/raw/ShotQA_meta/meta.jsonl
ShotQA_meta/sft.json
ShotQA_meta/grpo.json
data_shotqa/                  # or data/raw/data_shotqa/
ShotBench/image/              # or data/raw/ShotBench/image/
```

Generated CSV files are written to `csv/`, text manifests to `txt/`, and
processed split/cache files to `data/processed/`. The source datasets can stay
in their existing root-level locations or be placed under `data/raw/`; the
scripts resolve either layout. The repository does not contain the datasets or
the image files.

## Environment setup

Create and activate a virtual environment, then install the pinned base
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts also import packages that are not currently listed in
`requirements.txt`. Install them before running the full pipeline:

```bash
pip install scikit-learn torch transformers
```

The embedding script uses Apple MPS when available, then CUDA, and otherwise
CPU. It downloads `google/siglip-base-patch16-224` from Hugging Face on its
first run.

## Pipeline order

To run the complete preparation pipeline from any working directory, invoke:

```bash
python scripts/prepare_data.py
```

The runner resolves the repository root itself and executes the steps below in
order. It stops after leakage-safe train/validation/test split generation; it
does not run SigLIP embedding extraction or baseline evaluation.

### 1. Download ShotBench annotations

```bash
python make_annotations.py
```

Downloads the `Vchitect/ShotBench` test split with the Hugging Face `datasets`
library and saves it as `csv/annotations.csv`.

### 2. Parse ShotQA labels

```bash
python build_labels.py
```

Reads `ShotQA_meta/sft.json`, parses question/options/answer conversations, and
writes:

- `csv/shotqa_labels.csv`: image, question, and resolved label rows.
- `txt/lista_shot_size.txt`: image paths associated with shot-size questions.

It also prints label counts, dimension coverage, overlap with ShotBench, and a
small sample from `grpo.json`.

### 3. Check metadata and overlap

These are diagnostic scripts and may be run independently after
`csv/shotqa_labels.csv` exists:

```bash
python check_meta.py
python check_overlap.py
```

`check_meta.py` compares ShotQA labels with `meta.jsonl`, checks whether labels
agree with the metadata `Frame Size`, counts films, and creates
`txt/lista_<dimension>_tar.txt` files for shot size, framing, and camera angle.

`check_overlap.py` checks whether ShotBench images or films also appear in the
ShotQA training labels. This is an early leakage/overlap diagnostic and expects
`csv/labels_shot_size.csv`.

### 4. Build consolidated dimension labels

```bash
python final_labels.py
```

Merges SFT labels with GRPO labels for the dimensions listed in `USE` (currently
`shot_size` and `framing`), joins film metadata, marks multi-label examples,
and writes:

- `csv/labels_shot_size.csv`
- `csv/labels_framing.csv`
- `txt/lista_extrair.txt`: the combined list of image files to extract/download.

To include camera angle, add `camera_angle` to `USE` in the script before
running it.

### 5. Verify the local data

After the requested images have been placed in `data_shotqa/` or
`data/raw/data_shotqa/`, run:

```bash
python verify.py
```

This reports how many labeled images exist on disk, the number of extracted
files, the number of requested files, and the shape/columns of the ShotBench
annotations.

### 6. Create leakage-safe train/validation/test splits

```bash
python build_dataset.py
```

For each dimension in `DIMS` (`shot_size` and `framing`), this script:

1. Resolves ShotBench answer letters into labels.
2. Keeps only exact category matches.
3. Removes multi-label training examples.
4. Removes ShotQA training images from films represented in ShotBench.
5. Splits the remaining ShotQA data into train/validation groups by film.
6. Writes `data/processed/splits/train_<dimension>.csv`,
   `val_<dimension>.csv`, and `test_<dimension>.csv`.

The validation split is 15% by film with `random_state=42`. Missing film
metadata is assigned its image stem as a private group.

### 7. Extract embeddings and evaluate the baseline

```bash
python embed_shotsize.py
```

Despite the general project theme, the current implementation loops over
`shot_size` only. It:

1. Loads train, validation, and test split CSV files.
2. Computes normalized SigLIP image embeddings.
3. Caches them as `data/processed/emb/train_shot_size.npy`,
`data/processed/emb/val_shot_size.npy`, and
`data/processed/emb/test_shot_size.npy`.
4. Measures nearest-neighbor similarity from test to train/validation images.
5. Selects logistic-regression `C` using validation macro-F1.
6. Fits on train plus validation data.
7. Prints strict and permissive accuracy, macro-F1, and a confusion matrix.

Existing cached embeddings can be reused; the script skips extraction when the
corresponding `.npy` file already exists.

## Shot-size SigLIP fine-tuning

After the split CSVs and referenced images exist, train a shot-size classifier
with:

```bash
python scripts/finetune_shotsize.py
```

The script reuses `data/processed/splits/train_shot_size.csv`,
`val_shot_size.csv`, and `test_shot_size.csv`; it never regenerates them. By
default (`--mode head`) SigLIP is frozen and only a linear classification head
is trained. To also train the final vision layers, use for example:

```bash
python scripts/finetune_shotsize.py --mode last_layers --unfreeze-last-layers 2 \
  --epochs 5 --batch-size 16 --learning-rate 1e-4
```

Important options include `--model-name`, `--epochs`, `--batch-size`,
`--learning-rate`, `--seed`, and `--output-dir`. The default output directory
is `data/processed/finetune_shotsize/` and contains the best validation
macro-F1 checkpoint, `class_mapping.json`, `history.json`, test predictions,
and `metrics.json`. Test metrics are computed once from the best validation
checkpoint on single-label test rows; multi-label test rows remain in the
prediction CSV and are reported as excluded from strict classification metrics.
The model and local images are not included in Git. The command downloads the
configured Hugging Face model if it is not already cached, so run it locally
after preparing the datasets.

## Current implementation status

Already implemented:

- ShotQA SFT conversation parsing.
- GRPO/SFT label consolidation for shot size and framing.
- ShotBench annotation download and conversion to CSV.
- Metadata and train/test overlap diagnostics.
- Film-grouped, leakage-aware split generation.
- SigLIP embedding extraction and caching.
- A shot-size logistic-regression baseline with near-duplicate reporting.

The tracked `emb/` files show that shot-size embedding extraction has been run
at least once. No trained classifier artifact or prediction file is currently
tracked.

## Remaining work

1. **Prepare the data locally:** obtain `ShotQA_meta`, `data_shotqa`, and
   `ShotBench/image`, then run the pipeline in the order above.
2. **Make the environment reproducible:** add `scikit-learn`, `torch`,
   `transformers`, and any platform-specific installation notes to the
   dependency setup.
3. **Run and record a complete baseline:** save the final class counts,
   evaluation metrics, confusion matrix, and near-duplicate statistics.
4. **Extend evaluation beyond shot size:** adapt the embedding/evaluation
   script to run framing and, if desired, camera angle.
5. **Add model outputs:** save the selected classifier, predictions, and metric
   JSON instead of printing metrics only.
6. **Add automated checks:** test label parsing, category filtering, film-level
   splitting, missing files, and multi-label handling.
7. **Clean project presentation:** expand `README.md` with this workflow,
   dataset acquisition instructions, expected directory layout, and baseline
   results; remove the tracked `.DS_Store`.

Until the ignored datasets and extracted images are available, the full
pipeline cannot be reproduced from a clean clone using only the tracked files.
