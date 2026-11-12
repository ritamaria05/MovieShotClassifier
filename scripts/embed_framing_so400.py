"""Train and evaluate a framing classifier on frozen SigLIP embeddings.
Select classifier settings by validation macro-F1, then fit on train+val.
Default backbone: google/siglip-so400m-patch14-384 (cache files get a model suffix).
Usage: python scripts/embed_framing_so400.py [--classifier logreg|mlp] [--root DIR] [--out DIR] [--model NAME]
Inputs and embedding caches default to the project root; output defaults to
ROOT/outputs. Relative overrides are resolved from the current directory.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from sklearn.linear_model import LogisticRegression
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
ap.add_argument('--model', default='google/siglip-so400m-patch14-384',
                help='HF vision-language model used to extract embeddings')
ap.add_argument('--classifier', choices=['logreg', 'mlp'], default='logreg',
                help='classifier head (default: logreg)')
ap.add_argument('--threads', type=int, default=0, help='torch CPU threads (0 = default)')
ap.add_argument('--limit', type=int, default=0, help='benchmark only: embed N images, no cache, no eval')
a = ap.parse_args()
root = a.root
out_dir = a.out if a.out is not None else root / 'outputs'
emb_dir = root / 'emb'
dev = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
NAME = a.model
# The legacy siglip-base-224 cache keeps its original file names; other models get a suffix
# so the existing cache is never overwritten.
TAG = '' if NAME == 'google/siglip-base-patch16-224' else '_' + NAME.split('/')[-1]
model = None
proc = None
emb_dir.mkdir(parents=True, exist_ok=True)
if a.threads:
    torch.set_num_threads(a.threads)
def load(p):
    im = Image.open(p)
    # Source frames are >=1372x694 (median 1920x956), larger than the 384px model input, so
    # the reduced JPEG decode (still >=448 on both sides) loses no detail the model can use.
    im.draft('RGB', (448, 448))
    return im.convert('RGB')
@torch.no_grad()
def embed(df, cache, bs=32):
    global model, proc
    if cache.exists():
        return np.load(cache)
    if model is None:
        model = AutoModel.from_pretrained(NAME).eval().to(dev)
        proc = AutoImageProcessor.from_pretrained(NAME)
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
if a.limit:
    import time
    df = pd.read_csv(root / 'splits' / 'val_framing.csv').head(a.limit)
    t = time.time()
    embed(df, emb_dir / '_bench.npy')
    print(f'{(time.time() - t) / a.limit:.2f} s/img')
    (emb_dir / '_bench.npy').unlink()
    raise SystemExit
for name in ['framing']:
    tr, va, te = [pd.read_csv(root / 'splits' / f'{s}_{name}.csv') for s in ['train', 'val', 'test']]
    Xtr, Xva, Xte = [embed(d, emb_dir / f'{s}_{name}{TAG}.npy') for d, s in [(tr, 'train'), (va, 'val'), (te, 'test')]]
    # near-duplicates: teste vs treino+val
    sim = (Xte @ np.vstack([Xtr, Xva]).T).max(1)
    print(f'\n=== {name} ===\nsimilaridade máx. teste->treino: mediana {np.median(sim):.3f} | '
          f'p95 {np.quantile(sim, .95):.3f} | >0.95: {(sim > .95).sum()} de {len(te)}')
    clean = sim <= 0.95
    if a.classifier == 'logreg':
        grid = [{'C': c} for c in [0.1, 1, 10, 100, 1000]]
        def make_classifier(**params):
            return LogisticRegression(max_iter=3000, class_weight='balanced', **params)
    else:
        grid = MLP_GRID
        make_classifier = make_mlp
    scores = []
    for params in grid:
        clf = make_classifier(**params).fit(Xtr, tr['label'])
        score = f1_score(va['label'], clf.predict(Xva), average='macro')
        scores.append(score)
        print(f'{params}: val macro-F1 {score:.3f}')
    best_params = grid[int(np.argmax(scores))]
    print('melhores parametros:', best_params)
    clf = make_classifier(**best_params) \
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
    classifier_tag = 'logreg' if a.classifier == 'logreg' else 'MLP'
    cm.to_csv(out_dir / f'confusion_matrix_{name}_{classifier_tag}{TAG}.csv', index_label='actual')