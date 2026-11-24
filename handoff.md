# Handoff: MovieShotClassifier framing experiments

## Context

- Project: `C:\Users\hugsi\Mestrado\4.1_VC\MovieShotClassifier`.
- WSL Ubuntu environment: `source /home/hugo/.venv/movieshot/bin/activate`.
- User is investigating cinematic framing classification and prefers explanations grounded in project evidence. The dataset was explicitly left unmodified during the labeling inspection.
- WSL previously reported CUDA unavailable. A full SigLIP fine-tuning run on CPU took many hours.

## Current results

The strict test set contains 391 single-label images; another 54 test images have multiple labels. Comparisons below are for the single-label subset:

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Frozen base SigLIP + MLP | 79.28% | 77.87% |
| Fine-tuned base SigLIP + MLP, seed 42 | 80.05% | 78.99% |
| Frozen SO400M + MLP | 78.52% | 76.54% |
| Frozen base SigLIP + balanced MLP | 78.01% | 76.88% |
| Frozen SO400M + balanced MLP | approximately 77.5% | approximately 76.3% |
| Frozen base SigLIP + logistic regression | 76.21% | 75.10% |
| Frozen SO400M + logistic regression | 76.98% | 75.44% |

Fine-tuned metrics and classification report are in `outputs\finetune_framing\seed_42\metrics.json`; confusion matrix and predictions are in the same folder. The frozen baseline is `outputs\confusion_matrix_framing_MLP.csv`.

The fine-tuning run's best validation macro-F1 was 0.7364 at epoch 2. Early stopping triggered after epoch 5, with train macro-F1 0.9608 versus validation 0.7245, suggesting overfitting. This is a single-seed result, not a proven accuracy ceiling. The frozen baseline refits on train+val, whereas the fine-tuned model uses train and validation for checkpoint selection; procedures differ.

## Implementation and artifacts

- `scripts\finetune_framing.py`: trains a 256-hidden-unit MLP head on frozen base SigLIP, then fine-tunes the last two vision layers with lower backbone LR. Uses validation macro-F1, early stopping, per-seed outputs, test confusion matrix and strict/lenient metrics. The user ran `scripts\test_finetune_framing.py` and reported two tests passed.
- `scripts\eval_balanced_framing.py`: evaluates class-weighted scikit-learn MLP on cached embeddings; `--model so400m` selects SO400M cache. Model-specific outputs preserve the base result.
- `scripts\embed_framing.py`: frozen-base SigLIP framing baseline. `scripts\embed_framing_so400.py` supports MLP and logistic regression.
- Base and SO400M framing embeddings are already cached under `emb\`; avoid recomputing without a reason.
- `tests\framing2.md` and the experiment log contain project notes. User may have moved or edited files; inspect current state before modifying anything.

## Dataset observations

- Existing splits: 6,043 train, 955 validation, 445 test. Train includes 458 over-the-shoulder examples versus 1,100 single examples.
- `scripts\build_dataset.py` excludes multi-label training examples and uses film-grouped train/validation splitting. Test labels are from ShotBench. The original `labels_framing.csv` contains 9,204 records, including 2,206 multi-label records before split/leakage filters.
- A read-only audit found no missing referenced images, duplicate image stems between splits, train/validation film overlap, or split-label mismatches against available source annotations. A 14-image validation sample revealed ambiguous candidates but no confirmed labeling error; source agreement does not prove correctness.
- Dataset schemas contain image paths, framing labels and descriptive metadata, but no explicit structured person counts, bounding boxes or shoulder/pose coordinates were found. The pixels and framing labels contain indirect information; explicit spatial features would need separate detection/annotation.
- Multi-labeling was discussed as a possible next experiment, but not implemented. Check annotation completeness: unlisted labels may not be confirmed negatives. Use separate splits and film leakage checks, tune thresholds on validation, compare proper multi-label metrics and a top-1 baseline; avoid modifying the original dataset.

## Cautions and next steps

- Do not infer that 80% is a hard ceiling or that ambiguous labels are definitively wrong.
- Repeated comparisons on the same test set can bias model selection. Use film-grouped validation for future decisions and preferably a fresh held-out test for final assessment.
- Potential work discussed: assess annotation completeness; try a frozen-embedding multi-label head; evaluate person detector or pose features; run another fine-tuning seed only if the user accepts CPU cost. No new experiment was requested in the latest exchange.
- Inspect git status and existing user changes before editing; do not overwrite existing outputs or dataset files.
