import sys

import cv2
import numpy as np

# Fixed seed so k-means gives the same dominant colors for the same frame.
cv2.setRNGSeed(0)


def crop_letterbox(image_bgr, threshold=16, max_fraction=0.35):
    """Remove black letterbox (top/bottom) and pillarbox (left/right) bars.

    A row/column counts as a bar if its mean grayscale value is below
    `threshold`. At most `max_fraction` of each side is removed, so dark
    scenes are not cropped away. Returns the original image if nothing
    sensible remains.
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

    # Fully black frame (fade) or too little left -> keep the original
    if cropped.shape[0] < 0.2 * h or cropped.shape[1] < 0.2 * w:
        return image_bgr
    return cropped


def name_rgb_color(rgb):
    r, g, b = (int(c) for c in rgb)
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = (int(c) for c in hsv)

    if v < 50:
        return "Black"
    if s < 35:
        if v > 220:
            return "White"
        if v > 150:
            return "Light gray"
        if v > 75:
            return "Gray"
        return "Dark gray"
    if h < 8 or h >= 172:
        return "Red"
    if h < 23:
        return "Orange"
    if h < 38:
        return "Yellow"
    if h < 78:
        return "Green"
    if h < 105:
        return "Cyan"
    if h < 135:
        return "Blue"
    if h < 155:
        return "Purple"
    return "Magenta"


def _to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


def analyze_color(image_bgr, k=3):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mean_saturation = float(s.mean()) / 255
    # Same brightness definition as light_analysis.py (grayscale mean)
    mean_brightness = float(gray.mean()) / 255
    neutral_ratio = float((s < 30).mean())

    all_pixels = image_bgr.reshape(-1, 3).astype(np.float32)
    channel_mean = all_pixels.mean(axis=1)

    valid = (channel_mean > 20) & (channel_mean < 235)
    pixels = all_pixels[valid]

    # Very dark / very bright frames (fades, flashes): fall back to all pixels
    if len(pixels) < k:
        pixels = all_pixels

    pixels = pixels[::max(1, len(pixels) // 100_000)]

    mean_rgb = [int(c) for c in pixels.mean(axis=0)[::-1]]

    dominant_colors = []
    lab_centers = []

    if len(pixels) >= k:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten(), minlength=k)
        order = np.argsort(counts)[::-1]

        for index in order:
            if counts[index] == 0:
                continue
            b, g, r = np.clip(np.round(centers[index]), 0, 255).astype(np.uint8)
            rgb = [int(r), int(g), int(b)]

            lab_center = cv2.cvtColor(
                np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB
            )[0][0].astype(float)

            lab_centers.append(lab_center)
            dominant_colors.append({
                "name": name_rgb_color(rgb),
                "rgb": rgb,
                "hex": _to_hex(rgb),
                "percentage": float(round(100 * counts[index] / counts.sum(), 1)),
            })

    return {
        "mean_saturation": round(mean_saturation, 3),
        "mean_brightness": round(mean_brightness, 3),
        "neutral_ratio": round(neutral_ratio, 3),
        "mean_rgb": mean_rgb,
        "mean_hex": _to_hex(mean_rgb),
        "dominant_colors": dominant_colors,
        "lab_centers": lab_centers,  # numpy arrays: internal use only
    }


def analyze_tonal_style(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Same contrast definition and thresholds as light_analysis.py
    contrast = float(gray.std()) / 127.5
    mean_bgr = image_bgr.reshape(-1, 3).mean(axis=0)
    red_blue_difference = float(mean_bgr[2] - mean_bgr[0])

    if contrast < 0.15:
        contrast_level = "Low contrast"
    elif contrast < 0.35:
        contrast_level = "Medium contrast"
    else:
        contrast_level = "High contrast"

    if red_blue_difference > 12:
        temperature = "Warm"
    elif red_blue_difference < -12:
        temperature = "Cool"
    else:
        temperature = "Neutral"

    return {
        "contrast": contrast_level,
        "contrast_score": round(contrast, 3),
        "temperature": temperature,
        "red_blue_difference": round(red_blue_difference, 1),
    }


def merge_similar_colors(dominant_colors, lab_centers, threshold=25):
    merged = []

    for color, lab in zip(dominant_colors, lab_centers):
        for existing in merged:
            distance = np.linalg.norm(lab - existing["lab"])

            if distance < threshold:
                total = existing["percentage"] + color["percentage"]

                existing_rgb = np.array(existing["rgb"], dtype=float)
                current_rgb = np.array(color["rgb"], dtype=float)

                new_rgb = np.round(
                    (existing_rgb * existing["percentage"]
                     + current_rgb * color["percentage"]) / total
                ).astype(int)
                existing["rgb"] = [int(c) for c in new_rgb]
                existing["percentage"] = float(round(total, 1))
                existing["hex"] = _to_hex(existing["rgb"])
                existing["name"] = name_rgb_color(existing["rgb"])
                break
        else:
            merged.append({
                "name": color["name"],
                "rgb": list(color["rgb"]),
                "hex": color["hex"],
                "percentage": color["percentage"],
                "lab": lab,
            })

    for color in merged:
        color.pop("lab")

    return merged


def accent_colors(image_bgr, k=6, sat_min=100, val_min=90, merge_threshold=20,
                  min_frame_share=0.5):
    """Vivid colors that k-means on the whole frame drowns out (lamps, gels, neon, costumes).

    Only saturated, not-too-dark pixels are clustered, in Lab space so that
    distances follow perceived color. Each accent reports its share of the
    whole frame; accents below `min_frame_share` percent are dropped.
    Returns (accents, chromatic_share) where chromatic_share is the fraction
    of the frame that counts as vivid.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] >= sat_min) & (hsv[..., 2] >= val_min)
    chromatic_share = float(mask.mean())

    pixels = image_bgr[mask]
    if len(pixels) < k:
        return [], chromatic_share
    pixels = pixels[::max(1, len(pixels) // 100_000)]

    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)

    colors, labs = [], []
    for index in np.argsort(counts)[::-1]:
        if counts[index] == 0:
            continue
        lab_c = np.clip(np.round(centers[index]), 0, 255).astype(np.uint8)
        b, g, r = cv2.cvtColor(lab_c.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0][0]
        rgb = [int(r), int(g), int(b)]
        labs.append(lab_c.astype(float))
        colors.append({
            "name": name_rgb_color(rgb),
            "rgb": rgb,
            "hex": _to_hex(rgb),
            # share of the WHOLE frame, so it is comparable with the dominant colors
            "percentage": float(round(100 * chromatic_share * counts[index] / counts.sum(), 1)),
        })

    merged = merge_similar_colors(colors, labs, threshold=merge_threshold)
    return [c for c in merged if c["percentage"] >= min_frame_share], chromatic_share


def get_color_features(image_bgr, k=3, merge_threshold=25, min_percentage=8,
                       remove_letterbox=True):
    """Single entry point for the demo. Returns a JSON-serializable dict."""
    if remove_letterbox:
        image_bgr = crop_letterbox(image_bgr)

    analysis = analyze_color(image_bgr, k=k)
    tonal = analyze_tonal_style(image_bgr)

    merged = merge_similar_colors(
        analysis["dominant_colors"], analysis["lab_centers"], threshold=merge_threshold
    )
    merged = [c for c in merged if c["percentage"] >= min_percentage]
    accents, chromatic_share = accent_colors(image_bgr)

    return {
        "accent_colors": accents,
        "chromatic_share": round(chromatic_share, 3),
        "saturation": analysis["mean_saturation"],
        "brightness": analysis["mean_brightness"],
        "neutral_ratio": analysis["neutral_ratio"],
        "mean_rgb": analysis["mean_rgb"],
        "mean_hex": analysis["mean_hex"],
        "contrast": tonal["contrast"],
        "contrast_score": tonal["contrast_score"],
        "temperature": tonal["temperature"],
        "dominant_colors": merged,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python color_analysis.py /path/to/image.jpg")

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    result = get_color_features(image)

    print("Saturation:", round(result["saturation"] * 100, 1), "%")
    print("Brightness:", round(result["brightness"] * 100, 1), "%")
    print("Neutral pixels:", round(result["neutral_ratio"] * 100, 1), "%")
    print("Contrast:", result["contrast"])
    print("Temperature:", result["temperature"])
    print("Dominant Colors:")
    for color in result["dominant_colors"]:
        print(
            f'{color["name"]}: '
            f'{color["percentage"]}% '
            f'{color["hex"]}'
            f' (RGB: {color["rgb"]})'
        )
    print(f'Accent Colors (vivid pixels = {round(result["chromatic_share"] * 100, 1)}% of the frame):')
    for color in result["accent_colors"]:
        print(f'{color["name"]}: {color["percentage"]}% of frame {color["hex"]} (RGB: {color["rgb"]})')