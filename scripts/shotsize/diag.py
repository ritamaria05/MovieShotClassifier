import json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scripts.paths import shot_size_runs

runs = [str(path) for path in shot_size_runs()]
preds, rows = [], []
for r in runs:
    hist = json.load(open(f"{r}/history.json"))
    best, best_epoch = -np.inf, None
    for h in hist:                      # mesma regra do script (min_delta=0.001)
        if h["val_macro_f1"] > best + 0.001:
            best, best_epoch = h["val_macro_f1"], h["epoch"]
    m = json.load(open(f"{r}/metrics.json"))
    rows.append({"run": Path(r).name, "epochs_run": len(hist), "best_epoch": best_epoch,
                 "val_f1": m["best_val_macro_f1"], "test_f1": m["macro_f1"],
                 "gap_val_minus_test": m["best_val_macro_f1"] - m["macro_f1"]})
    p = pd.read_csv(f"{r}/predictions_test.csv")
    preds.append(p)
df = pd.DataFrame(rows)
print(df.round(4).to_string(index=False))
print(f"\nval F1: {df.val_f1.mean():.4f} ± {df.val_f1.std(ddof=1):.4f} | "
      f"test F1: {df.test_f1.mean():.4f} ± {df.test_f1.std(ddof=1):.4f} | "
      f"gap médio val-test: {df.gap_val_minus_test.mean():+.4f}")

# diversidade entre seeds (só linhas single-label)
ref = preds[0]
single = ~ref["label"].str.contains(",")
P = np.stack([p.set_index("path").loc[ref["path"], "prediction"].to_numpy()[single.to_numpy()] for p in preds])
y = ref.loc[single, "label"].to_numpy()
agree = np.mean([np.mean(P[i] == P[j]) for i, j in itertools.combinations(range(len(P)), 2)])
vote = pd.DataFrame(P).mode(axis=0).iloc[0].to_numpy()
accs = (P == y).mean(axis=1)
print(f"\nconcordância média entre seeds (previsões idênticas): {agree:.3f}")
print(f"accuracy individual: {accs.mean():.4f} | voto maioritário: {(vote == y).mean():.4f} "
      "(diagnóstico, não uses o teste para escolher configurações)")