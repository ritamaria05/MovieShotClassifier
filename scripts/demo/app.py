"""Visual demo server for the shot-size and framing models.

Run from the repository root:
    python -m pip install -r demo_requirements.txt
    python -m uvicorn scripts.demo.app:app --port 8000
Then open http://127.0.0.1:8000

Everything shown comes from this project's own artefacts:
  * shot size and framing: Hugging Face checkpoints
  * heatmap:   occlusion of the model's own predicted-class probability
  * neighbours: cosine similarity in the cached frozen-SigLIP train+val embeddings
"""
from __future__ import annotations

import io
import json
import logging
import re
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import httpx2
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel

from movieshot.models import (
    SHOT_SIZE_ORDER, download_framing_checkpoint, download_shot_size_checkpoints,
    load_framing_model, load_shot_size_checkpoints, model_probabilities, select_device,
)
from scripts.paths import DATA_PROCESSED_DIR, ROOT, SPLITS_DIR
from scripts.color.color_analysis import get_color_features
from scripts.lighting.light_analysis import analyze_lighting, crop_letterbox

STATIC = Path(__file__).parent / "static"
BASE_NAME = "google/siglip-base-patch16-224"
DATASET_REPO = "hugomelo17/MovieShotClassifier-dataset"
DATASET_REVISION = "dccc8f570b92363138aa909af0c251e6c4646eed"
GRID = 8
CONFUSABLE = [{"3 shot", "group shot"}, {"over the shoulder", "2 shot"}]   # from the confusion matrices

STATE: dict = {}
LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------- loading
def reference_image_path(path: str) -> str | None:
    """Map a ShotQA split path to its uploaded dataset image."""
    parts = str(path).replace("\\", "/").split("/")
    if ".." in parts or len(parts) < 2 or parts[-2] != "data_shotqa":
        return None
    prefix = [part for part in parts[:-2] if part and part != "."]
    project_names = {ROOT.name.casefold(), "movieshotclassifier"}
    project_path = not prefix or prefix[-1].casefold() in project_names
    raw_path = prefix == ["data", "raw"] or (
        len(prefix) >= 3 and prefix[-2:] == ["data", "raw"]
        and prefix[-3].casefold() in project_names
    )
    name = parts[-1]
    if not (project_path or raw_path) or not re.fullmatch(
        r"[A-Z0-9][A-Za-z0-9._-]*\.(?:jpg|jpeg|png)", name, re.IGNORECASE
    ) or name[0] not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return None
    folder = "0-H" if name[0] in "0123456789ABCDEFGH" else "I-Z"
    return f"images/{folder}/{name}"


def load_index(dim: str):
    """Train+val frozen embeddings and image paths for nearest-neighbour lookup."""
    candidates = [(SPLITS_DIR, DATA_PROCESSED_DIR / "emb"), (ROOT / "splits", ROOT / "emb")]
    for splits, emb in candidates:
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
                "remote_paths": [reference_image_path(p) for p in df["path"]]}
    print(f"[demo] {dim}: no cached embeddings found, neighbours disabled")
    return None


@asynccontextmanager
async def lifespan(app):
    device = select_device()
    print(f"[demo] device: {device}")
    try:
        checkpoints = download_shot_size_checkpoints()
        models, proc, label_to_id = load_shot_size_checkpoints(checkpoints, device)
        models = [m.to(device).eval() for m in models]                     # same device as the inputs
        classes = [c for c, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])]
        order = [c for c in SHOT_SIZE_ORDER if c in classes]
        missing = [c for c in classes if c not in order]
        if missing:                                                         # spelling/case mismatch would crash /analyze
            print(f"[demo] WARNING: classes not in DEFAULT_ORDER, appended at the end: {missing}")
            order += missing
        print(f"[demo] shot size classes: {classes}")
        STATE["shot_size"] = {"models": models, "proc": proc, "classes": classes, "device": device,
                              "order": order, "index": load_index("shot_size")}
    except (Exception, SystemExit) as e:                                    # keep the other feature usable
        traceback.print_exc()
        print(f"[demo] shot size unavailable: {e!r}")
    try:
        checkpoint = download_framing_checkpoint()
        model, proc, classes = load_framing_model(checkpoint, device)
        STATE["framing"] = {"models": [model], "proc": proc, "classes": classes, "device": device,
                            "order": classes, "index": load_index("framing"),
                            "metrics_path": checkpoint.parent / "metrics.json"}
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
def model_probs(feat: dict, images: list[Image.Image], models=None) -> np.ndarray:
    """(n_models, n_images, n_classes) softmax probabilities."""
    return model_probabilities(models or feat["models"], feat["proc"], feat["device"], images)


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


def look(image: Image.Image) -> dict | None:
    """Colour + lighting features from the two OpenCV scripts (no model involved)."""
    try:
        small = image.copy()
        small.thumbnail((1280, 1280))                 # speed; the statistics barely change
        bgr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)   # PIL is RGB, the scripts expect BGR
        color = get_color_features(bgr)
        light = analyze_lighting(bgr)
        gray = cv2.cvtColor(crop_letterbox(bgr), cv2.COLOR_BGR2GRAY)
        hist = np.histogram(gray, bins=32, range=(0, 256))[0].astype(float)
        hist = hist / hist.max() if hist.max() > 0 else hist      # tallest bar = 1, for the chart
        return {
            "histogram": hist.round(3).tolist(),
            "lighting": light["lighting"],
            "brightness": light["brightness"],
            "contrast": light["contrast"],            # one contrast definition (both scripts now agree)
            "contrast_score": light["contrast_score"],
            "shadows": light["shadow_ratio"],
            "highlights": light["highlight_ratio"],
            "dynamic_range": light["dynamic_range"],
            "saturation": color["saturation"],
            "neutral": color["neutral_ratio"],
            "temperature": color["temperature"],
            "mean_hex": color["mean_hex"],
            "palette": color["dominant_colors"],
            "accents": color["accent_colors"],                 # vivid colors, % of the whole frame
            "vivid": color["chromatic_share"],
        }
    except Exception:                                 # never let this break the model result
        traceback.print_exc()
        return None


def reliability(label: str, metrics_path: Path | None):
    if metrics_path is None or not metrics_path.exists():
        return None
    rep = json.loads(metrics_path.read_text())["test_classification_report"].get(label)
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
    if key not in ("shot_size", "framing"):
        raise HTTPException(404)
    index = STATE.get(key, {}).get("index")
    if not index or not 0 <= i < len(index["remote_paths"]):
        raise HTTPException(404)
    remote_path = index["remote_paths"][i]
    if remote_path is None:
        raise HTTPException(404)
    try:
        path = Path(hf_hub_download(DATASET_REPO, remote_path, repo_type="dataset", revision=DATASET_REVISION))
    except (OSError, httpx2.HTTPError) as exc:
        LOGGER.exception("Could not download reference image %s", remote_path)
        raise HTTPException(503, "Could not retrieve the reference image from Hugging Face.") from exc
    return FileResponse(path)


def classify(feature: str, feat: dict, image: Image.Image) -> dict:
    """Everything the page shows for one model: label, probabilities, heatmap, neighbours."""
    per_model = model_probs(feat, [image])[:, 0]
    mean = per_model.mean(0)
    classes, order = feat["classes"], feat["order"]
    ranked = np.argsort(-mean)
    top, second = int(ranked[0]), int(ranked[1])
    label, label2 = classes[top], classes[second]
    margin = float(mean[top] - mean[second])
    note = None
    if feature == "framing" and margin < 0.3:
        note = f"Borderline between “{label}” and “{label2}”, which are ambiguous in this picture."
        if {label, label2} in CONFUSABLE:
            note += " This is also a pair the model often confuses."
    elif feature == "shot_size" and abs(order.index(label) - order.index(label2)) == 1 and margin < 0.25:
        note = f"Borderline between “{label}” and “{label2}”, which are neighbouring sizes."
    return {
        "feature": feature, "label": label, "label2": label2, "confidence": float(mean[top]),
        "order": order, "probs": [float(mean[classes.index(c)]) for c in order],
        "votes": int((per_model.argmax(1) == top).sum()), "n_models": int(len(per_model)),
        "heat": occlusion(feat, image, top), "neighbours": neighbours(feat, image, feature),
        "reliability": reliability(label, feat.get("metrics_path")) if feature == "framing" else None, "note": note,
    }


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    """One click computes everything; the page only decides which part to show."""
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    result = {"look": look(image)}                    # no shared model state, so outside the lock
    with LOCK:                                        # one request at a time on the shared models
        for key in ("shot_size", "framing"):          # a model that is not loaded gives None
            result[key] = classify(key, STATE[key], image) if key in STATE else None
    return result