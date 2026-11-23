"""Visual demo server for the shot-size and framing models.

Run from the repository root:
    python -m pip install -r requirements.txt
    python -m uvicorn scripts.demo.app:app --port 8000
Then open http://127.0.0.1:8000

Everything shown comes from this project's own artefacts:
  * shot size: the fine-tuned seed ensemble (data/processed/finetune_shotsize_protocol/seed*)
  * framing:   outputs/finetune_framing/seed_42/best_finetune.pt
  * heatmap:   occlusion of the model's own predicted-class probability
  * neighbours: cosine similarity in the cached frozen-SigLIP train+val embeddings
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts"):          # `scripts.paths` and plain `paths` are both used in the repo
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.paths import DATA_PROCESSED_DIR, SPLITS_DIR                    # noqa: E402
from scripts.shotsize.predict_shotsize import DEFAULT_ORDER, load_models    # noqa: E402

STATIC = Path(__file__).parent / "static"
SS_RUNS = str(ROOT / "data/processed/finetune_shotsize_protocol/seed*")
SS_SINGLE = ROOT / "data/processed/finetune_shotsize"      # default output dir of a single training run
FRAMING_CKPT = ROOT / "outputs/finetune_framing/seed_42/best_finetune.pt"
FRAMING_METRICS = ROOT / "outputs/finetune_framing/seed_42/metrics.json"
BASE_NAME = "google/siglip-base-patch16-224"
GRID = 8
CONFUSABLE = [{"3 shot", "group shot"}, {"over the shoulder", "2 shot"}]   # from the confusion matrices

STATE: dict = {}
LOCK = threading.Lock()


# ---------------------------------------------------------------- loading
def load_framing(device):
    spec = importlib.util.spec_from_file_location("finetune_framing", ROOT / "scripts/framing/finetune_framing.py")
    ff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ff)
    ck = torch.load(FRAMING_CKPT, map_location="cpu", weights_only=False)
    backbone = AutoModel.from_pretrained(ck["model_name"])
    model = ff.FramingClassifier(backbone, backbone.config.vision_config.hidden_size,
                                 ck["hidden_size"], len(ck["label_to_id"]))
    model.load_state_dict(ck["model_state"])
    if hasattr(model.backbone, "text_model"):
        del model.backbone.text_model
    classes = [c for c, _ in sorted(ck["label_to_id"].items(), key=lambda kv: kv[1])]
    proc = AutoImageProcessor.from_pretrained(ck["model_name"])
    return model.to(device).eval(), proc, classes


def load_index(dim: str):
    """Train+val frozen embeddings and image paths for nearest-neighbour lookup."""
    candidates = [(ROOT / "splits", ROOT / "emb", ROOT), (SPLITS_DIR, DATA_PROCESSED_DIR / "emb", Path("."))]
    for splits, emb, base in candidates:
        files = [splits / f"{s}_{dim}.csv" for s in ("train", "val")] + [emb / f"{s}_{dim}.npy" for s in ("train", "val")]
        if not all(f.exists() for f in files):
            continue
        frames = [pd.read_csv(f) for f in files[:2]]
        X = np.vstack([np.load(f) for f in files[2:]])
        df = pd.concat(frames, ignore_index=True)
        if len(df) != len(X):
            print(f"[demo] {dim}: embeddings/CSV mismatch, neighbours disabled")
            return None
        return {"X": X, "labels": df["label"].astype(str).tolist(),
                "paths": [str(base / p) for p in df["path"]]}
    print(f"[demo] {dim}: no cached embeddings found, neighbours disabled")
    return None


@asynccontextmanager
async def lifespan(app):
    from scripts.shotsize.finetune_shotsize import select_device
    device = select_device()
    print(f"[demo] device: {device}")
    try:
        import glob
        runs = sorted(Path(p) for p in glob.glob(SS_RUNS) if (Path(p) / "best_model.pt").is_file())
        if not runs and (SS_SINGLE / "best_model.pt").is_file():
            runs = [SS_SINGLE]                                              # a single run is enough for the demo
        if not runs:
            raise FileNotFoundError(f"no best_model.pt under {SS_RUNS} or {SS_SINGLE}")
        print(f"[demo] shot size runs: {[r.name for r in runs]}")
        models, proc, label_to_id, _, _ = load_models(runs)
        classes = [c for c, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])]
        STATE["shot_size"] = {"models": models, "proc": proc, "classes": classes, "device": device,
                              "order": [c for c in DEFAULT_ORDER if c in classes], "index": load_index("shot_size")}
    except (Exception, SystemExit) as e:                                    # keep the other feature usable
        traceback.print_exc()
        print(f"[demo] shot size unavailable: {e!r}")
    try:
        model, proc, classes = load_framing(device)
        STATE["framing"] = {"models": [model], "proc": proc, "classes": classes, "device": device,
                            "order": classes, "index": load_index("framing")}
    except (Exception, SystemExit) as e:
        traceback.print_exc()
        print(f"[demo] framing unavailable: {e!r}")
    if any(s["index"] for s in STATE.values()):
        base = AutoModel.from_pretrained(BASE_NAME).eval().to(device)
        if hasattr(base, "text_model"):
            del base.text_model
        STATE["base"] = (base, AutoImageProcessor.from_pretrained(BASE_NAME))
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def model_probs(feat: dict, images: list[Image.Image], models=None) -> np.ndarray:
    """(n_models, n_images, n_classes) softmax probabilities."""
    models = models or feat["models"]
    inputs = {k: v.to(feat["device"]) for k, v in feat["proc"](images=images, return_tensors="pt").items()}
    return np.stack([torch.softmax(m(inputs).float(), 1).cpu().numpy() for m in models])


def occlusion(feat: dict, image: Image.Image, cls: int, batch: int = 12) -> list[list[float]]:
    """How much the predicted-class probability drops when each grid cell is hidden."""
    small = image.copy()
    small.thumbnail((640, 640))
    w, h = small.size
    fill = tuple(int(v) for v in np.asarray(small).reshape(-1, 3).mean(0))
    models = feat["models"][:1]                       # one model keeps this to a few seconds on CPU
    base = model_probs(feat, [small], models)[0, 0, cls]
    variants = []
    for r in range(GRID):
        for c in range(GRID):
            im = small.copy()
            ImageDraw.Draw(im).rectangle([c * w // GRID, r * h // GRID,
                                          (c + 1) * w // GRID - 1, (r + 1) * h // GRID - 1], fill=fill)
            variants.append(im)
    drops = np.concatenate([base - model_probs(feat, variants[i:i + batch], models)[0, :, cls]
                            for i in range(0, len(variants), batch)])
    drops = np.clip(drops, 0, None)
    drops = drops / drops.max() if drops.max() > 0 else drops
    return drops.reshape(GRID, GRID).round(3).tolist()


@torch.no_grad()
def neighbours(feat: dict, image: Image.Image, key: str, k: int = 4) -> list[dict]:
    index = feat["index"]
    if not index or "base" not in STATE:
        return []
    base, proc = STATE["base"]
    inputs = {a: b.to(feat["device"]) for a, b in proc(images=[image], return_tensors="pt").items()}
    f = base.get_image_features(**inputs)
    f = f if torch.is_tensor(f) else f.pooler_output
    q = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()[0]
    sims = index["X"] @ q
    top = np.argsort(-sims)[:k]
    return [{"url": f"/thumb/{key}/{i}", "label": index["labels"][i], "sim": round(float(sims[i]), 3)} for i in top]


def reliability(label: str):
    if not FRAMING_METRICS.exists():
        return None
    rep = json.loads(FRAMING_METRICS.read_text())["test_classification_report"].get(label)
    return {"precision": rep["precision"], "recall": rep["recall"]} if rep else None


# ---------------------------------------------------------------- routes
@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/status")
def status():
    return {"features": [k for k in ("shot_size", "framing") if k in STATE]}


@app.get("/thumb/{key}/{i}")
def thumb(key: str, i: int):
    index = STATE.get(key, {}).get("index")
    if not index or not 0 <= i < len(index["paths"]) or not Path(index["paths"][i]).is_file():
        raise HTTPException(404)
    return FileResponse(index["paths"][i])


@app.post("/analyze")
def analyze(feature: str = Form(...), file: UploadFile = File(...)):
    feat = STATE.get(feature)
    if feat is None:
        raise HTTPException(400, f"Feature '{feature}' is not loaded. Check the server log.")
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    with LOCK:                                        # one request at a time on the shared models
        per_model = model_probs(feat, [image])[:, 0]
        mean = per_model.mean(0)
        classes, order = feat["classes"], feat["order"]
        ranked = np.argsort(-mean)
        top, second = int(ranked[0]), int(ranked[1])
        label, label2 = classes[top], classes[second]
        margin = float(mean[top] - mean[second])
        note = None
        if feature == "framing" and {label, label2} in CONFUSABLE and margin < 0.3:
            note = f"“{label}” and “{label2}” are the pair this model mixes up most often."
        elif feature == "shot_size" and abs(order.index(label) - order.index(label2)) == 1 and margin < 0.25:
            note = f"Borderline between “{label}” and “{label2}”, which are neighbouring sizes."
        return {
            "feature": feature, "label": label, "label2": label2, "confidence": float(mean[top]),
            "order": order, "probs": [float(mean[classes.index(c)]) for c in order],
            "votes": int((per_model.argmax(1) == top).sum()), "n_models": int(len(per_model)),
            "heat": occlusion(feat, image, top), "neighbours": neighbours(feat, image, feature),
            "reliability": reliability(label) if feature == "framing" else None, "note": note,
        }
