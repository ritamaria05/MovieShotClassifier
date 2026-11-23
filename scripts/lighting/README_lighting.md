# Lighting analysis

Classifies movie-frame lighting (default dimensions: `lighting_type` and
`lighting_condition`) with three feature sets, so you can see what each one adds:

| Feature set | What it is |
| --- | --- |
| `embeddings` | frozen SigLIP image embeddings (same model/cache style as framing/shot size) |
| `handcrafted` | ~40 interpretable descriptors: tone, contrast/edge softness, light direction, colour/warmth |
| `hybrid` | both concatenated |

A majority-class baseline is always reported. Existing shot-size / framing files
are not touched; everything new lives in `scripts/lighting/`.

## Files

- `discover_lighting.py` – read-only; lists lighting categories/labels in ShotQA SFT, GRPO and ShotBench.
- `build_lighting_dataset.py` – film-grouped, leakage-safe splits (same rules as `build_dataset.py`).
- `extract_lighting_features.py` – handcrafted descriptors (crops letterbox bars first).
- `embed_lighting.py` – embedding cache, validation-based selection, train+val refit, test metrics.

## Run order (from the repository root)

```bash
python scripts/lighting/discover_lighting.py             # 1. confirm category names
python scripts/lighting/build_lighting_dataset.py        # 2. splits (+ txt/lista_lighting_extrair.txt)
python scripts/lighting/extract_lighting_features.py     # 3. handcrafted features
python scripts/lighting/embed_lighting.py --dim lighting_type
python scripts/lighting/embed_lighting.py --dim lighting_condition
```

Useful options: `--classifier mlp`, `--feature-sets handcrafted`,
`--model google/siglip-so400m-patch14-384` (separate cache suffix), and
`--dim NAME=category` on the builder if the category names differ.

Outputs: `data/processed/splits/*_lighting_*.csv`,
`data/processed/lighting/features_*.csv`, `data/processed/emb/*_lighting_*.npy`,
and `outputs/lighting/` (metrics JSON, confusion matrices, predictions).

## Things to check

- **Images must be extracted.** `txt/lista_extrair.txt` only covers shot size and
  framing. Extract the images in `txt/lista_lighting_extrair.txt` into
  `data_shotqa/`; the builder excludes (and reports) labelled images missing on disk.
- **Category names are an assumption.** I could not inspect your local files, so
  `lighting type` / `lighting condition` are my guess based on the ShotBench
  naming. `discover_lighting.py` shows the real ones. If a category yields no
  ShotBench rows the builder warns and skips it.
- **Class counts may be small.** Check the per-split class table the builder
  prints; very rare classes make macro-F1 noisy.
- **Handcrafted features are proxies.** They cannot tell which light is the key
  light or distinguish day from night by themselves; treat their results as an
  interpretable baseline, not ground truth about the lighting setup.
- **Test-set reuse.** Choose configurations with validation numbers; the test
  numbers are one look at the held-out set.
- Run `pip install -r requirements.txt` first (OpenCV, tqdm, scikit-learn, torch, transformers are used).
