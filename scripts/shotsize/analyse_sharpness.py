from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


CLASS_ORDER = [
    "extreme close up",
    "close up",
    "medium close up",
    "medium",
    "medium wide",
    "wide",
    "extreme wide",
]

output_dir = Path("data/processed/sharpness_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

results = {}

for split in ("train", "val", "test"):
    frame = pd.read_csv(f"data/processed/shot_features_{split}.csv")

    # Exclude multi-label rows from this shot-size analysis
    single = frame.loc[~frame["label"].str.contains(",")]
    single = single.copy()
    single["laplacian_sharpness_log"] = np.log1p(
        single["laplacian_sharpness"]
    )

    present = [label for label in CLASS_ORDER if label in single["label"].unique()]
    grouped = [
        single.loc[single["label"] == label, "laplacian_sharpness_log"]
        for label in present
    ]

    plt.figure(figsize=(12, 6))
    plt.boxplot(grouped, tick_labels=present, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("log(1 + Laplacian variance)")
    plt.title(f"Sharpness by shot-size class — {split}")
    plt.tight_layout()
    plt.savefig(output_dir / f"sharpness_{split}.png", dpi=200)
    plt.savefig(output_dir / f"sharpness_{split}.svg")
    plt.close()

    stats = (
        single.groupby("label")["laplacian_sharpness"]
        .agg(["count", "median", "mean", "std"])
        .reindex(present)
        .reset_index()
    )
    stats["median_log"] = np.log1p(stats["median"])
    stats.to_csv(output_dir / f"sharpness_stats_{split}.csv", index=False)
    results[split] = stats

print("Results saved to:", output_dir)

for split, stats in results.items():
    print(f"\n{split}")
    print(stats.to_string(index=False))