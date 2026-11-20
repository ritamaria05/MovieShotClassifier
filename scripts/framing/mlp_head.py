"""Shared MLP head and hyperparameter grid for frozen embeddings."""

from sklearn.neural_network import MLPClassifier


MLP_GRID = [
    {'hidden_layer_sizes': h, 'alpha': a}
    for h in [(256,), (512,), (512, 128)]
    for a in [1e-4, 1e-2]
]


def make_mlp(**params):
    return MLPClassifier(max_iter=300, early_stopping=True, random_state=42, **params)
