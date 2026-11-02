import os, numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

dev = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
NAME = 'google/siglip-base-patch16-224'
model = AutoModel.from_pretrained(NAME).eval().to(dev)
proc = AutoImageProcessor.from_pretrained(NAME)
os.makedirs('emb', exist_ok=True)

def load(p):
    im = Image.open(p)
    im.draft('RGB', (448, 448))      # descodifica o JPEG já em tamanho reduzido
    return im.convert('RGB')

@torch.no_grad()
def embed(df, cache, bs=32):
    if os.path.exists(cache):
        return np.load(cache)
    paths, out = df['path'].tolist(), []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i in tqdm(range(0, len(paths), bs), desc=os.path.basename(cache)):
            imgs = list(ex.map(load, paths[i:i+bs]))
            x = proc(images=imgs, return_tensors='pt').to(dev)
            f = model.get_image_features(**x)
            f = f if torch.is_tensor(f) else f.pooler_output
            out.append(f.float().cpu())
    e = torch.nn.functional.normalize(torch.cat(out), dim=-1).numpy()
    np.save(cache, e)
    return e

for name in ['shot_size']:
    tr, va, te = [pd.read_csv(f'splits/{s}_{name}.csv') for s in ['train', 'val', 'test']]
    Xtr, Xva, Xte = [embed(d, f'emb/{s}_{name}.npy') for d, s in [(tr, 'train'), (va, 'val'), (te, 'test')]]

    # near-duplicates: teste vs treino+val
    sim = (Xte @ np.vstack([Xtr, Xva]).T).max(1)
    print(f'\n=== {name} ===\nsimilaridade máx. teste->treino: mediana {np.median(sim):.3f} | '
          f'p95 {np.quantile(sim, .95):.3f} | >0.95: {(sim > .95).sum()} de {len(te)}')
    clean = sim <= 0.95

    scores = {}
    for C in [0.1, 1, 10, 100]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(Xtr, tr['label'])
        scores[C] = f1_score(va['label'], clf.predict(Xva), average='macro')
        print(f'C={C}: val macro-F1 {scores[C]:.3f}')
    best_C = max(scores, key=scores.get)
    print('melhor C:', best_C)

    clf = LogisticRegression(C=best_C, max_iter=3000, class_weight='balanced') \
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
    print(pd.DataFrame(confusion_matrix(te.label[s], pred[s], labels=labels), index=labels, columns=labels))