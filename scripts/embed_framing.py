"""Train and evaluate an MLP framing classifier on frozen SigLIP embeddings.

Select the MLP's settings by validation macro-F1, then fit on train+val.

Usage: python scripts/embed_framing.py [--root DIR] [--out DIR]
Inputs and embedding caches default to the project root; output defaults to
ROOT/outputs. Relative overrides are resolved from the current directory.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from mlp_head import MLP_GRID, make_mlp
from paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, default=ROOT,
                help='directory containing splits and emb (default: project root)')
ap.add_argument('--out', type=Path, default=None,
                help='output directory (default: ROOT/outputs)')
a = ap.parse_args()
root = a.root
out_dir = a.out if a.out is not None else root / 'outputs'
emb_dir = root / 'emb'

dev = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
NAME = 'google/siglip-base-patch16-224'
model = AutoModel.from_pretrained(NAME).eval().to(dev)
proc = AutoImageProcessor.from_pretrained(NAME)
emb_dir.mkdir(parents=True, exist_ok=True)

def load(p):
    im = Image.open(p)
    im.draft('RGB', (448, 448))      # descodifica o JPEG já em tamanho reduzido
    return im.convert('RGB')

@torch.no_grad()
def embed(df, cache, bs=32):
    if cache.exists():
        return np.load(cache)
    paths, out = [root / p for p in df['path']], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i in tqdm(range(0, len(paths), bs), desc=cache.name):
            imgs = list(ex.map(load, paths[i:i+bs]))
            x = proc(images=imgs, return_tensors='pt').to(dev)
            f = model.get_image_features(**x)
            f = f if torch.is_tensor(f) else f.pooler_output
            out.append(f.float().cpu())
    e = torch.nn.functional.normalize(torch.cat(out), dim=-1).numpy()
    np.save(cache, e)
    return e

for name in ['framing']:
    tr, va, te = [pd.read_csv(root / 'splits' / f'{s}_{name}.csv') for s in ['train', 'val', 'test']]
    Xtr, Xva, Xte = [embed(d, emb_dir / f'{s}_{name}.npy') for d, s in [(tr, 'train'), (va, 'val'), (te, 'test')]]

    # near-duplicates: teste vs treino+val
    sim = (Xte @ np.vstack([Xtr, Xva]).T).max(1)
    print(f'\n=== {name} ===\nsimilaridade máx. teste->treino: mediana {np.median(sim):.3f} | '
          f'p95 {np.quantile(sim, .95):.3f} | >0.95: {(sim > .95).sum()} de {len(te)}')
    clean = sim <= 0.95

    scores = []
    for params in MLP_GRID:
        clf = make_mlp(**params).fit(Xtr, tr['label'])
        score = f1_score(va['label'], clf.predict(Xva), average='macro')
        scores.append(score)
        print(f'{params}: val macro-F1 {score:.3f}')
    best_params = MLP_GRID[int(np.argmax(scores))]
    print('melhores parametros:', best_params)

    clf = make_mlp(**best_params) \
        .fit(np.vstack([Xtr, Xva]), pd.concat([tr['label'], va['label']]))
    pred = clf.predict(Xte)

    single = ~te['multi'].values
    lenient = np.array([p in set(l.split(', ')) for p, l in zip(pred, te['label'])])
    for tag, m in [('todo o teste', np.ones(len(te), bool)), ('sem near-dups', clean)]:
        s = single & m
        print(f'[{tag}] estrita: acc {accuracy_score(te.label[s], pred[s]):.3f} '
              f'macro-F1 {f1_score(te.label[s], pred[s], average="macro"):.3f} (n={s.sum()}) | '
              f'permissiva: acc {lenient[m].mean():.3f}')
    s = single
    labels = sorted(set(te.label[s]))
    cm = pd.DataFrame(
        confusion_matrix(te.label[s], pred[s], labels=labels),
        index=labels,
        columns=labels,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cm.to_csv(out_dir / f'confusion_matrix_{name}_MLP.csv', index_label='actual')