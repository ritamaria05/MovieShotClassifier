# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paths import DATA_PROCESSED_DIR, resolve_image_path, split_file

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
cascade = cv2.CascadeClassifier(cascade_path)


def extract_features(path: str) -> tuple[float, float, float, float, float]:
    image = cv2.imread(str(resolve_image_path(path)))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    if len(faces):
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face_ratio = h / gray.shape[0]
        face_area_ratio = (w * h) / (gray.shape[0] * gray.shape[1])
        headroom = y / gray.shape[0]
    else:
        face_ratio = np.nan
        face_area_ratio = np.nan
        headroom = np.nan

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    height, width = gray.shape
    center = laplacian[
        height // 4:3 * height // 4,
        width // 4:3 * width // 4,
    ]
    laplacian_center = float(center.var())

    return (
        float(face_ratio),
        float(face_area_ratio),
        float(headroom),
        sharpness,
        laplacian_center,
    )


def extract_split(split: str) -> None:
    input_path = split_file(f"{split}_shot_size.csv")

    frame = pd.read_csv(input_path)

    columns = [
        "face_ratio",
        "face_area_ratio",
        "headroom",
        "laplacian_sharpness",
        "laplacian_center",
    ]

    rows = [
        extract_features(path)
        for path in tqdm(frame["path"], desc=f"Extracting {split}")
    ]

    features = pd.DataFrame(rows, columns=columns)
    output = pd.concat([frame, features], axis=1)
    output.to_csv(
        DATA_PROCESSED_DIR / f"shot_features_{split}.csv",
        index=False,
    )


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        extract_split(split)