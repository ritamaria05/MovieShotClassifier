"""
THIS SCRIPT DIDNT HELP MUCH
Hard-parameter-sharing MLP (shared layer + shot_size head + framing head) on frozen SigLIP
embeddings, with masked loss, vs single-task MLPs with the same architecture/protocol.
Few images carry both labels, so each image contributes only to the heads it has a label for.
Epoch chosen by val macro-F1 (mean over tasks for MTL); test reported at that epoch.
Usage: python scripts/mtl_hard_sharing.py [--seeds 5] [--root DIR] [--out DIR]
By default, inputs come from the project root and output goes to its outputs directory.
Relative --root and --out paths are resolved from the current working directory.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from paths import ROOT

DIMS = ['shot_size', 'framing']
def load(root):
    data = {}
    for s in ['train', 'val', 'test']:
        rows = {}
        for d in DIMS:
            df = pd.read_csv(root / 'splits' / f'{s}_{d}.csv')
            X = np.load(root / 'emb' / f'{s}_{d}.npy')
            assert len(df) == len(X)
            for i, p in enumerate(df['path']):
                r = rows.setdefault(p, {'x': X[i]})
                r[d] = df['label'].iloc[i]
                r[d + '_multi'] = bool(df['multi'].iloc[i]) if 'multi' in df else False
        data[s] = rows
    return data
def tensors(rows, vocab):
    X = np.stack([r['x'] for r in rows.values()]).astype(np.float32)
    Y = {d: np.array([vocab[d].get(r.get(d), -1) if not r.get(d + '_multi') else -1
                      for r in rows.values()]) for d in DIMS}
    return torch.tensor(X), {d: torch.tensor(v) for d, v in Y.items()}
class Net(nn.Module):
    def __init__(self, nc, tasks, h=512, p=0.3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(768, h), nn.ReLU(), nn.Dropout(p))
        self.heads = nn.ModuleDict({d: nn.Linear(h, nc[d]) for d in tasks})
    def forward(self, x):
        z = self.body(x)
        return {d: hd(z) for d, hd in self.heads.items()}
def predict(net, X):
    net.eval()
    with torch.no_grad():
        return {d: o.argmax(1).numpy() for d, o in net(X).items()}
def score(net, X, Y, tasks):
    pr, out = predict(net, X), {}
    for d in tasks:
        m = (Y[d] >= 0).numpy()
        out[d] = (accuracy_score(Y[d][m], pr[d][m]), f1_score(Y[d][m], pr[d][m], average='macro'))
    return out
def train(tasks, tr, va, te, vocab, seed, epochs=60, lr=1e-3, wd=1e-2, bs=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    nc = {d: len(vocab[d]) for d in tasks}
    net = Net(nc, tasks)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    (Xt, Yt), (Xv, Yv), (Xe, Ye) = tr, va, te
    lossf = {}
    for d in tasks:
        y = Yt[d][Yt[d] >= 0]
        w = len(y) / (nc[d] * torch.bincount(y, minlength=nc[d]).clamp(min=1))
        lossf[d] = nn.CrossEntropyLoss(weight=w.float(), ignore_index=-1)   # masked loss
    best, best_val = None, -1
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            out = net(Xt[idx])
            loss = 0
            for d in tasks:
                if (Yt[d][idx] >= 0).any():
                    loss = loss + lossf[d](out[d], Yt[d][idx])
            if isinstance(loss, int):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
        v = score(net, Xv, Yv, tasks)
        vm = np.mean([v[d][1] for d in tasks])
        if vm > best_val:
            best_val, best = vm, (ep + 1, v, score(net, Xe, Ye, tasks))
    return best
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--root', type=Path, default=ROOT,
                    help='directory containing splits and emb (default: project root)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output directory (default: ROOT/outputs)')
    a = ap.parse_args()
    out = a.out if a.out is not None else a.root / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    data = load(a.root)
    vocab = {d: {l: i for i, l in enumerate(sorted(
        {r[d] for r in data['train'].values() if d in r and not r[d + '_multi']}))} for d in DIMS}
    sets = {s: tensors(data[s], vocab) for s in data}
    # drop rows with no usable label for any task (e.g. only multi-label)
    # and, for single-task runs, rows without that task's label are masked anyway.
    res = {}
    for seed in range(a.seeds):
        runs = {'mtl': train(DIMS, sets['train'], sets['val'], sets['test'], vocab, seed)}
        for d in DIMS:
            runs[f'single_{d}'] = train([d], sets['train'], sets['val'], sets['test'], vocab, seed)
        for k, (ep, v, t) in runs.items():
            for d in t:
                res.setdefault(f'{k}|{d}', []).append({'seed': seed, 'epoch': ep, 'val_f1': v[d][1],
                                                       'test_acc': t[d][0], 'test_macro_f1': t[d][1]})
        print(f'seed {seed} done', flush=True)
    summary = {}
    print('\nmodel | task: test strict acc / macro-F1 (mean +- std over seeds)')
    for k, v in res.items():
        f = np.array([r['test_macro_f1'] for r in v]); acc = np.array([r['test_acc'] for r in v])
        summary[k] = {'test_acc_mean': acc.mean(), 'test_acc_std': acc.std(),
                      'test_macro_f1_mean': f.mean(), 'test_macro_f1_std': f.std(), 'runs': v}
        print(f'{k}: acc {acc.mean():.3f}+-{acc.std():.3f}  macro-F1 {f.mean():.3f}+-{f.std():.3f}')
    with (out / 'mtl_hard_sharing_metrics.json').open('w') as f:
        json.dump(summary, f, indent=2, default=float)