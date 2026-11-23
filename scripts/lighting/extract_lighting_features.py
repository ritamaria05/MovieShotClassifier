"""Extract interpretable lighting descriptors from movie frames.

Features (all computed after cropping black letterbox bars, which would
otherwise inflate the "dark" statistics of every widescreen frame):

  tone         mean/std/percentiles of luminance, dynamic range, dark / bright /
               clipped fractions, histogram entropy (low-key vs high-key)
  contrast     gradient (edge) strength and local contrast: hard light gives
               sharp shadow edges, soft light gives gradual transitions
  direction    3x3 luminance grid, left-right / top-bottom asymmetry,
               centre-vs-border difference (vignette / spotlight)
  colour       mean saturation, colourfulness, warm-cool balance overall and
               separately in shadows and highlights (e.g. teal/orange grading)

These are PROXIES. They do not know which light is the key light or whether the
scene is day or night; they are meant as a transparent baseline and as extra
inputs next to SigLIP embeddings.

Usage:
  python scripts/lighting/extract_lighting_features.py [--dims lighting_type lighting_condition]
Output: data/processed/lighting/features_<dim>_<split>.csv  (split CSV + features)
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paths import DATA_PROCESSED_DIR, SPLITS_DIR

LIGHTING_DIR = DATA_PROCESSED_DIR / "lighting"
TARGET_WIDTH = 384
EPS = 1e-6


def crop_black_bars(bgr: np.ndarray, threshold: int = 12) -> np.ndarray:
    """Remove near-black rows/columns at the frame border (letterboxing)."""
    peak = bgr.max(axis=2)
    rows = np.flatnonzero(peak.max(axis=1) > threshold)
    cols = np.flatnonzero(peak.max(axis=0) > threshold)
    if len(rows) == 0 or len(cols) == 0:
        return bgr  # genuinely black frame: keep as is
    cropped = bgr[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]
    if cropped.shape[0] * cropped.shape[1] < 0.5 * bgr.shape[0] * bgr.shape[1]:
        return bgr  # implausible crop
    return cropped


def warmth(rgb: np.ndarray, mask: np.ndarray) -> float:
    """log(R/B) over masked pixels; 0 when the mask is empty."""
    if not mask.any():
        return 0.0
    r, b = rgb[..., 0][mask].mean(), rgb[..., 2][mask].mean()
    return float(np.log((r + EPS) / (b + EPS)))


def extract_features(path: str) -> dict[str, float]:
    bgr = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    bgr = crop_black_bars(bgr)
    height, width = bgr.shape[:2]
    if width > TARGET_WIDTH:
        bgr = cv2.resize(bgr, (TARGET_WIDTH, max(1, int(height * TARGET_WIDTH / width))),
                         interpolation=cv2.INTER_AREA)

    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    f: dict[str, float] = {}

    # ---- tone
    p05, p25, p50, p75, p95 = np.percentile(luma, [5, 25, 50, 75, 95])
    f.update(luma_mean=float(luma.mean()), luma_std=float(luma.std()),
             luma_p05=float(p05), luma_p25=float(p25), luma_p50=float(p50),
             luma_p75=float(p75), luma_p95=float(p95),
             dynamic_range=float(p95 - p05),
             frac_dark=float((luma < 0.15).mean()),
             frac_bright=float((luma > 0.85).mean()),
             frac_crushed=float((luma < 0.02).mean()),
             frac_clipped=float((luma > 0.98).mean()))
    hist, _ = np.histogram(luma, bins=32, range=(0.0, 1.0))
    prob = hist / max(1, hist.sum())
    prob = prob[prob > 0]
    f["hist_entropy"] = float(-(prob * np.log2(prob)).sum())

    # ---- contrast / edge softness
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    f.update(grad_mean=float(magnitude.mean()),
             grad_p95=float(np.percentile(magnitude, 95)),
             frac_strong_edges=float((magnitude > 0.5).mean()))
    small = cv2.resize(luma, (96, max(1, int(96 * luma.shape[0] / luma.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    mean = cv2.blur(small, (7, 7))
    local_var = np.clip(cv2.blur(small * small, (7, 7)) - mean * mean, 0, None)
    f["local_contrast"] = float(np.sqrt(local_var).mean())

    # ---- direction
    grid = cv2.resize(luma, (3, 3), interpolation=cv2.INTER_AREA)
    for r in range(3):
        for c in range(3):
            f[f"grid_{r}{c}"] = float(grid[r, c])
    left, right = grid[:, 0].mean(), grid[:, 2].mean()
    top, bottom = grid[0].mean(), grid[2].mean()
    f["asym_lr"] = float((right - left) / (right + left + EPS))
    f["asym_tb"] = float((top - bottom) / (top + bottom + EPS))
    border = (grid.sum() - grid[1, 1]) / 8.0
    f["center_minus_border"] = float(grid[1, 1] - border)
    f["max_cell_minus_mean"] = float(grid.max() - grid.mean())

    # ---- colour
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    f.update(sat_mean=float(hsv[..., 1].mean() / 255.0), sat_std=float(hsv[..., 1].std() / 255.0))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    f["colorfulness"] = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    f["warm_cool"] = warmth(rgb, np.ones(luma.shape, bool))
    f["warm_shadows"] = warmth(rgb, luma <= p05 + EPS) if p05 < p95 else 0.0
    f["warm_highlights"] = warmth(rgb, luma >= p95 - EPS) if p05 < p95 else 0.0
    f["warm_split"] = f["warm_highlights"] - f["warm_shadows"]
    return f


def extract_split(dim: str, split: str, workers: int) -> Path | None:
    source = SPLITS_DIR / f"{split}_{dim}.csv"
    if not source.exists():
        print(f"Skipping {source} (not found)")
        return None
    frame = pd.read_csv(source)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(tqdm(pool.map(extract_features, frame["path"]),
                         total=len(frame), desc=f"{dim}/{split}"))
    features = pd.DataFrame(rows)
    clash = set(features.columns) & set(frame.columns)
    if clash:
        raise ValueError(f"Feature names clash with split columns: {sorted(clash)}")
    LIGHTING_DIR.mkdir(parents=True, exist_ok=True)
    output = LIGHTING_DIR / f"features_{dim}_{split}.csv"
    pd.concat([frame, features], axis=1).to_csv(output, index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dims", nargs="+", default=["lighting_type", "lighting_condition"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for dim in args.dims:
        for split in ("train", "val", "test"):
            output = extract_split(dim, split, args.workers)
            if output:
                print("wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
