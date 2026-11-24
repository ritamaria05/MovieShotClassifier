# MovieShotClassifier WILL UPDATE

Analyze movie frames by shot size, framing, color and lighting.

## WSL setup

From a WSL terminal, activate your existing environment and enter the repository:

```bash
source ~/.venv/movieshot/bin/activate
cd /mnt/c/Users/hugsi/Mestrado/4.1_VC/MovieShotClassifier
python -m pip install -r requirements.txt
```

Use the actual path to your environment or repository if different. Run the
commands below from the repository root, using `-m` and no `.py` suffix.
Training requires the local ShotQA/ShotBench metadata and images described in
[PROJECT_GUIDE.md](PROJECT_GUIDE.md).

Place the datasets under `data/raw/data_shotqa/`, `data/raw/ShotBench/`, and
`data/raw/ShotQA_meta/`. Existing split CSVs containing the old root-level
image paths still work; they do not need to be regenerated after the move.

## Prepare data and train

If you already have split CSVs and their referenced images, skip preparation.
Otherwise, after obtaining the data:

```bash
python -m scripts.prepare_data
```

Train a shot-size model (default: frozen SigLIP backbone, five epochs):

```bash
python -m scripts.shotsize.finetune_shotsize
```

It saves the best checkpoint to `data/processed/models/finetune_shotsize/best_model.pt`,
alongside metrics and predictions. This can take substantial time and memory;
the first run downloads the pretrained SigLIP model. For framing training:

```bash
python -m scripts.framing.finetune_framing
```

That saves the default seed's checkpoint to
`data/processed/models/finetune_framing/seed_42/best_finetune.pt`.

The optional baseline command `python -m scripts.shotsize.embed_shotsize`
evaluates logistic regression but does not save a trained classifier.

## Run the demo

```bash
python -m uvicorn scripts.demo.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Restart the server after training to load the
checkpoint. Color and lighting work without trained classifiers. For shot size,
the demo prefers the local single checkpoint above, or the available local
ensemble in `data/processed/models/finetune_shotsize/seed*/`. For framing it
prefers the local seed 42 checkpoint. Older checkpoint locations are also
supported.

If either model is missing locally, the demo downloads that model from
[hugomelo17/MovieShotClassifier](https://huggingface.co/hugomelo17/MovieShotClassifier)
and stores it in the Hugging Face cache (normally `~/.cache/huggingface/hub/`).
The shot-size fallback downloads all five seeds; framing uses `framing_model.pt`.
The fallback uses a pinned repository revision for reproducibility. The first
download requires internet and roughly 4.9 GB for these six checkpoints, plus
space for the pretrained Hugging Face backbone and image processor. Subsequent
runs reuse the cache. To use the cache offline, start with `HF_HUB_OFFLINE=1`.
Download errors are reported in the server log; the other features remain
available. Framing's per-class reliability and nearest-neighbour examples
require local metrics and datasets/embedding caches respectively.

The shot-size command `python -m scripts.shotsize.predict_shotsize` uses the
same local-first fallback; `--single` downloads only the first seed when no
local model is available. Explicit `--runs` directories remain local-only.

## Artifact locations

New split CSVs and embedding caches go to `data/processed/splits/` and
`data/processed/emb/`. Model checkpoints and their run metadata, predictions
and logits stay together under `data/processed/models/finetune_shotsize/` or
`data/processed/models/finetune_framing/`. Other reports go to `outputs/csv/`,
`outputs/json/`, or `outputs/txt/`, retaining experiment subfolders. For
example, ensemble reports go to `outputs/csv/shotsize_ensemble/ensemble/`
and `outputs/json/shotsize_ensemble/ensemble/`, with probabilities in
`data/processed/shotsize_ensemble/ensemble/`. A custom output directory outside
`outputs/` keeps its files together.
Readers prefer these locations and fall back to older root-level `splits/`
and `emb/` files. Keep embedding caches paired with the split CSVs used to
produce them: matching row counts alone does not establish matching order.
Existing artifacts are not moved automatically.
