import sys

import cv2
import numpy as np


def crop_letterbox(image_bgr, threshold=16, max_fraction=0.35):
    """Remove black letterbox (top/bottom) and pillarbox (left/right) bars.

    Same helper as in color_analysis.py, duplicated so each script runs alone.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    row_dark = gray.mean(axis=1) < threshold
    col_dark = gray.mean(axis=0) < threshold

    def leading(mask, limit):
        n = 0
        while n < limit and mask[n]:
            n += 1
        return n

    max_rows = int(h * max_fraction)
    max_cols = int(w * max_fraction)

    top = leading(row_dark, max_rows)
    bottom = leading(row_dark[::-1], max_rows)
    left = leading(col_dark, max_cols)
    right = leading(col_dark[::-1], max_cols)

    cropped = image_bgr[top:h - bottom, left:w - right]

    if cropped.shape[0] < 0.2 * h or cropped.shape[1] < 0.2 * w:
        return image_bgr
    return cropped


def analyze_lighting(image_bgr, remove_letterbox=True):
    """Single entry point for the demo. Returns a JSON-serializable dict."""
    if remove_letterbox:
        image_bgr = crop_letterbox(image_bgr)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    brightness_normalized = float(gray.mean()) / 255
    contrast_normalized = float(gray.std()) / 127.5

    percentile_05 = float(np.percentile(gray, 5))
    percentile_95 = float(np.percentile(gray, 95))
    dynamic_range = percentile_95 - percentile_05

    shadow_ratio = float(np.mean(gray < 64))
    highlight_ratio = float(np.mean(gray > 192))

    if brightness_normalized < 0.35:
        lighting = "Low-key"
    elif brightness_normalized < 0.65:
        lighting = "Mid-key"
    else:
        lighting = "High-key"

    if contrast_normalized < 0.15:
        contrast_level = "Low contrast"
    elif contrast_normalized < 0.35:
        contrast_level = "Medium contrast"
    else:
        contrast_level = "High contrast"

    return {
        "lighting": lighting,
        "brightness": round(brightness_normalized, 3),
        "contrast": contrast_level,
        "contrast_score": round(contrast_normalized, 3),
        "shadow_ratio": round(shadow_ratio, 3),
        "highlight_ratio": round(highlight_ratio, 3),
        "dynamic_range": round(dynamic_range, 1),
        "p05": round(percentile_05, 1),
        "p95": round(percentile_95, 1),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python light_analysis.py /path/to/image.jpg")

    image = cv2.imread(sys.argv[1])

    if image is None:
        raise SystemExit(f"Could not read image: {sys.argv[1]}")

    result = analyze_lighting(image)

    print("Lighting:", result["lighting"])
    print("Brightness:", round(result["brightness"] * 100, 1), "%")
    print("Contrast:", result["contrast"])
    print("Shadows:", round(result["shadow_ratio"] * 100, 1), "%")
    print("Highlights:", round(result["highlight_ratio"] * 100, 1), "%")
    print("Dynamic range:", result["dynamic_range"])