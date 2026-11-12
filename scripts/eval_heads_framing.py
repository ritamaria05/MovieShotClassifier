"""
Compare heads on cached frozen SigLIP embeddings (framing and/or shot_size).
Selection: GroupKFold by film over train+val (macro-F1); final fit on train+val; test with the
same strict/lenient metrics as embed_framing.py. Never recomputes embeddings.
Usage: python scripts/eval_heads_framing.py [--dims framing shot_size] [--root DIR] [--out DIR]
Inputs default to the project root; output defaults to ROOT/outputs. Relative
overrides are resolved from the current directory.
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from mlp_head import MLP_GRID, make_mlp
from paths import ROOT

warnings.filterwarnings('ignore')
HEADS = {
    'majority': (lambda **p: DummyClassifier(strategy='most_frequent'), [{}]),
    'logreg': (lambda **p: LogisticRegression(max_iter=3000, class_weight='balanced', **p),
               [{'C': c} for c in [0.1, 1, 3, 10, 30, 100]]),
    'svm_rbf': (lambda **p: SVC(kernel='rbf', class_weight='balanced', **p),
                [{'C': c, 'gamma': g} for c in [1, 3, 10, 30] for g in ['scale', 1, 2]]),
    'knn': (lambda **p: KNeighborsClassifier(metric='cosine', **p),
            [{'n_neighbors': k, 'weights': w} for k in [5, 10, 20, 40] for w in ['uniform', 'distance']]),
    'mlp': (make_mlp, MLP_GRID),
}
js = lambda p: {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
def run(dim, root, out, folds):
    ld = lambda s: pd.read_csv(root / 'splits' / f'{s}_{dim}.csv')
    tr, va, te = ld('train'), ld('val'), ld('test')
    Xtr, Xva, Xte = [np.load(root / 'emb' / f'{s}_{dim}.npy') for s in ['train', 'val', 'test']]
    for d, X in [(tr, Xtr), (va, Xva), (te, Xte)]:
        assert len(d) == len(X), f'{dim}: embedding/CSV row mismatch'
    X = np.vstack([Xtr, Xva])
    y = pd.concat([tr['label'], va['label']]).values
    film = pd.concat([tr['film'], va['film']]).reset_index(drop=True)
    groups = np.where(film.isna(), [f'img{i}' for i in range(len(film))], film.fillna(''))
    splits = list(GroupKFold(n_splits=folds).split(X, y, groups))
    cv, best = {}, {}
    for name, (make, grid) in HEADS.items():
        rows = []
        for p in grid:
            s = [f1_score(y[b], make(**p).fit(X[a], y[a]).predict(X[b]), average='macro') for a, b in splits]
            rows.append({'params': js(p), 'cv_macro_f1': float(np.mean(s)), 'cv_std': float(np.std(s))})
            print(f'[{dim}] {name} {p}: CV macro-F1 {np.mean(s):.3f} +- {np.std(s):.3f}', flush=True)
        i = max(range(len(grid)), key=lambda i: rows[i]['cv_macro_f1'])
        best[name] = (grid[i], rows[i]['cv_macro_f1'])
        cv[name] = rows
    selected = max((n for n in best if n != 'majority'), key=lambda n: best[n][1])
    single = ~te['multi'].values
    labels = sorted(set(te.label[single]) | set(y))
    metrics, preds = {}, {}
    for name, (p, cvf1) in best.items():
        pr = HEADS[name][0](**p).fit(X, y).predict(Xte)
        preds[name] = pr
        len_ok = np.array([a in set(l.split(', ')) for a, l in zip(pr, te['label'])])
        metrics[name] = {'best_params': js(p), 'cv_macro_f1': cvf1,
                         'test_strict_acc': accuracy_score(te.label[single], pr[single]),
                         'test_strict_macro_f1': f1_score(te.label[single], pr[single], average='macro'),
                         'test_n_strict': int(single.sum()), 'test_lenient_acc': float(len_ok.mean())}
        m = metrics[name]
        print(f"[{dim}] TEST {name}: strict acc {m['test_strict_acc']:.3f} macro-F1 "
              f"{m['test_strict_macro_f1']:.3f} lenient {m['test_lenient_acc']:.3f}")
    with (out / f'{dim}_heads_metrics.json').open('w') as f:
        json.dump({'selected': selected, 'metrics': metrics, 'cv_grid': cv}, f, indent=2)
    pd.DataFrame({'path': te['path'], 'label': te['label'], 'multi': te['multi'],
                  **{f'pred_{n}': p for n, p in preds.items()}}).to_csv(out / f'{dim}_heads_predictions.csv', index=False)
    sel = preds[selected]
    rep = classification_report(te.label[single], sel[single], digits=3, zero_division=0)
    (out / f'{dim}_report_{selected}.txt').write_text(rep, encoding='utf-8')
    cm = pd.DataFrame(confusion_matrix(te.label[single], sel[single], labels=labels), index=labels, columns=labels)
    cm.to_csv(out / f'{dim}_confusion_{selected}.csv')
    print(f'[{dim}] selected {selected}\n{rep}\n{cm}')
    bad = single & (sel != te.label.values)
    w = te[bad][['path', 'label']].copy()
    w['pred'] = sel[bad]
    focus = {'over the shoulder', '3 shot', 'group shot'} if dim == 'framing' else set()
    w['focus'] = w['label'].isin(focus) | w['pred'].isin(focus)
    w.sort_values(['focus', 'label'], ascending=[False, True]).to_csv(
        out / f'{dim}_misclassified_{selected}.csv', index=False)
    print(f'[{dim}] misclassified {len(w)} of {single.sum()}')
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dims', nargs='+', default=['framing', 'shot_size'])
    ap.add_argument('--root', type=Path, default=ROOT,
                    help='directory containing splits and emb (default: project root)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output directory (default: ROOT/outputs)')
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()
    out = a.out if a.out is not None else a.root / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    for d in a.dims:
        run(d, a.root, out, a.folds)