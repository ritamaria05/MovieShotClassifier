import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import eval_balanced_framing as experiment


class BalancedFramingTests(unittest.TestCase):
    def test_fit_uses_inverse_frequency_weights(self):
        labels = pd.Series(['a', 'a', 'a', 'b'])
        with patch.object(experiment, 'make_mlp') as make_mlp:
            experiment.fit_balanced(np.zeros((4, 2)), labels, {'alpha': 0.01})
        weights = make_mlp.return_value.fit.call_args.kwargs['sample_weight']
        np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2])

    def test_evaluate_selects_on_validation_and_saves_results(self):
        class FakeClassifier:
            def __init__(self, value):
                self.value = value

            def predict(self, features):
                if self.value == 1:
                    return np.where(features[:, 0] > 0, 'b', 'a')
                return np.full(len(features), 'a')

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'splits').mkdir()
            (root / 'emb').mkdir()
            for name in ('train', 'val', 'test'):
                data = {'path': ['a', 'b'], 'label': ['a', 'b'], 'multi': [False, False]}
                pd.DataFrame(data).to_csv(root / 'splits' / f'{name}_framing.csv', index=False)
                np.save(root / 'emb' / f'{name}_framing.npy', np.array([[-1., 0.], [1., 0.]]))
            calls = []

            def fit(features, labels, params):
                calls.append((len(labels), params))
                return FakeClassifier(params['value'])

            grid = [{'value': 0}, {'value': 1}]
            with patch.object(experiment, 'MLP_GRID', grid), patch.object(experiment, 'fit_balanced', side_effect=fit):
                experiment.evaluate(root, root / 'results')
            self.assertEqual(calls, [(2, grid[0]), (2, grid[1]), (4, grid[1])])
            with (root / 'results' / 'framing_MLP_balanced_metrics.json').open() as f:
                metrics = json.load(f)
            self.assertEqual(metrics['best_params'], {'value': 1})
            self.assertEqual(metrics['test_strict_accuracy'], 1.0)
            self.assertEqual(metrics['test_strict_macro_f1'], 1.0)
            matrix = pd.read_csv(root / 'results' / 'confusion_matrix_framing_MLP_balanced.csv', index_col='actual')
            np.testing.assert_array_equal(matrix.to_numpy(), [[1, 0], [0, 1]])

    def test_rejects_misaligned_embeddings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'splits').mkdir()
            (root / 'emb').mkdir()
            pd.DataFrame({'label': ['a', 'b']}).to_csv(root / 'splits' / 'train_framing.csv', index=False)
            np.save(root / 'emb' / 'train_framing.npy', np.zeros((1, 2)))
            with self.assertRaisesRegex(ValueError, 'one 2D embedding row'):
                experiment.load_split(root, 'train')

    def test_so400m_uses_separate_cache_and_output(self):
        class FakeClassifier:
            def predict(self, features):
                return np.where(features[:, 0] > 0, 'b', 'a')

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'splits').mkdir()
            (root / 'emb').mkdir()
            suffix = experiment.MODEL_SUFFIXES['so400m']
            for name in ('train', 'val', 'test'):
                pd.DataFrame({'label': ['a', 'b'], 'multi': [False, False]}).to_csv(
                    root / 'splits' / f'{name}_framing.csv', index=False)
                np.save(root / 'emb' / f'{name}_framing{suffix}.npy',
                        np.array([[-1., 0.], [1., 0.]]))
            with patch.object(experiment, 'MLP_GRID', [{}]), \
                    patch.object(experiment, 'fit_balanced', return_value=FakeClassifier()):
                experiment.evaluate(root, root / 'results', 'so400m')
            results = root / 'results'
            self.assertTrue((results / f'confusion_matrix_framing_MLP_balanced{suffix}.csv').is_file())
            with (results / f'framing_MLP_balanced{suffix}_metrics.json').open() as f:
                self.assertEqual(json.load(f)['embedding_model'], 'so400m')
            self.assertFalse((results / 'confusion_matrix_framing_MLP_balanced.csv').exists())


if __name__ == '__main__':
    unittest.main()
