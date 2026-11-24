"""Evaluate a class-weighted MLP on cached SigLIP framing embeddings.

Usage: python -m scripts.framing.eval_balanced_framing [--model base|so400m] [--root DIR] [--out DIR]

Uses the same MLP grid, seed, validation selection and train+validation refit
as embed_framing.py. Inverse-frequency sample weights are computed separately
for each fit. Relative paths are resolved from the current directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from scripts.framing.mlp_head import MLP_GRID, make_mlp
from scripts.paths import ROOT, embedding_file, prepare_report, split_file

MODEL_SUFFIXES = {'base': '', 'so400m': '_siglip-so400m-patch14-384'}


def load_split(root, split, suffix=''):
    frame = pd.read_csv(split_file(f'{split}_framing.csv', root))
    features = np.load(embedding_file(f'{split}_framing{suffix}.npy', root))
    if features.ndim != 2 or len(frame) != len(features):
        raise ValueError(f'{split}: expected one 2D embedding row per CSV row')
    if frame.empty:
        raise ValueError(f'{split}: split is empty')
    return frame, features


def fit_balanced(features, labels, params):
    weights = compute_sample_weight(class_weight='balanced', y=labels)
    return make_mlp(**params).fit(features, labels, sample_weight=weights)


def evaluate(root, out, model='base'):
    suffix = MODEL_SUFFIXES[model]
    tr, Xtr = load_split(root, 'train', suffix)
    va, Xva = load_split(root, 'val', suffix)
    te, Xte = load_split(root, 'test', suffix)

    scores = []
    for params in MLP_GRID:
        clf = fit_balanced(Xtr, tr['label'], params)
        score = f1_score(va['label'], clf.predict(Xva), average='macro')
        scores.append(float(score))
        print(f'{params}: val macro-F1 {score:.3f}', flush=True)

    best = int(np.argmax(scores))
    best_params = MLP_GRID[best]
    print(f'best parameters: {best_params}', flush=True)
    labels_train = pd.concat([tr['label'], va['label']], ignore_index=True)
    clf = fit_balanced(np.vstack([Xtr, Xva]), labels_train, best_params)
    pred = clf.predict(Xte)

    single = ~te['multi'].to_numpy()
    if not single.any():
        raise ValueError('test split has no single-label examples')
    true = te['label'].to_numpy()
    lenient = np.array([p in set(label.split(', ')) for p, label in zip(pred, true)])
    sim = (Xte @ np.vstack([Xtr, Xva]).T).max(axis=1)
    clean = sim <= 0.95
    labels = sorted(set(true[single]) | set(pred[single]))
    report = classification_report(true[single], pred[single], labels=labels,
                                   output_dict=True, zero_division=0)
    metrics = {
        'embedding_model': model,
        'best_params': best_params,
        'validation_macro_f1': scores[best],
        'validation_grid': [
            {'params': params, 'macro_f1': score}
            for params, score in zip(MLP_GRID, scores)
        ],
        'test_strict_accuracy': float(accuracy_score(true[single], pred[single])),
        'test_strict_macro_f1': float(f1_score(true[single], pred[single], average='macro')),
        'test_n_strict': int(single.sum()),
        'test_lenient_accuracy': float(lenient.mean()),
        'test_near_duplicates': int((~clean).sum()),
        'classification_report': report,
    }
    clean_single = single & clean
    if clean_single.any():
        metrics['test_clean_strict_accuracy'] = float(accuracy_score(true[clean_single], pred[clean_single]))
        metrics['test_clean_strict_macro_f1'] = float(f1_score(true[clean_single], pred[clean_single], average='macro'))
        metrics['test_clean_n_strict'] = int(clean_single.sum())
    else:
        print('No single-label test examples remain after near-duplicate filtering.', flush=True)
    if clean.any():
        metrics['test_clean_lenient_accuracy'] = float(lenient[clean].mean())
    else:
        print('No test examples remain after near-duplicate filtering.', flush=True)

    out.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(confusion_matrix(true[single], pred[single], labels=labels),
                          index=labels, columns=labels)
    matrix.to_csv(prepare_report(out, f'confusion_matrix_framing_MLP_balanced{suffix}.csv', root), index_label='actual')
    with prepare_report(out, f'framing_MLP_balanced{suffix}_metrics.json', root).open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"test strict accuracy {metrics['test_strict_accuracy']:.3f}, "
          f"macro-F1 {metrics['test_strict_macro_f1']:.3f}, "
          f"lenient accuracy {metrics['test_lenient_accuracy']:.3f}", flush=True)
    print(classification_report(true[single], pred[single], labels=labels, zero_division=0))
    print(f'Near-duplicates above 0.95: {(~clean).sum()} of {len(te)}')
    print(f"CSV saved to {prepare_report(out, f'confusion_matrix_framing_MLP_balanced{suffix}.csv', root)}")
    print(f"JSON saved to {prepare_report(out, f'framing_MLP_balanced{suffix}_metrics.json', root)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='directory containing splits and emb (default: project root)')
    parser.add_argument('--out', type=Path, default=None,
                        help='output directory (default: ROOT/outputs)')
    parser.add_argument('--model', choices=MODEL_SUFFIXES, default='base',
                        help='cached embedding model (default: base)')
    args = parser.parse_args()
    evaluate(args.root, args.out if args.out is not None else args.root / 'outputs', args.model)


if __name__ == '__main__':
    main()
