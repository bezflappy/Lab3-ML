from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    classification_metrics, compare_classical_models,
    train_linear_regression, train_logistic_classifier,
)
from run_experiments import prepare_data, run


class GradientTests(unittest.TestCase):
    def test_linear_first_step_by_hand(self):
        data = RegressionInput([[1], [2]], [3, 5], [[3]], [7], learning_rate=0.1, epochs=1)
        result = train_linear_regression(data)
        np.testing.assert_allclose(result.weights, [1.3])
        self.assertAlmostEqual(result.bias, 0.8)
        np.testing.assert_allclose(result.predictions, [4.7])
        np.testing.assert_allclose(result.loss_history, [(0.9**2+1.6**2)/2])

    def test_linear_recovers_known_coefficients(self):
        x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
        y = x @ [2, -3] + 4
        result = train_linear_regression(RegressionInput(x, y, x, y))
        np.testing.assert_allclose(result.weights, [2, -3], atol=1e-8)
        self.assertAlmostEqual(result.bias, 4)
        self.assertEqual(len(result.loss_history), 800)
        self.assertLess(result.loss_history[-1], 1e-12)

    def test_column_target_and_zero_feature(self):
        data = RegressionInput([[0], [0]], [[7], [7]], [[0]], [7])
        result = train_linear_regression(data)
        np.testing.assert_allclose(result.weights, [0])
        np.testing.assert_allclose(result.predictions, [7], atol=1e-8)

    def test_logistic_first_step_by_hand(self):
        data = ClassificationInput([[-1], [1]], [0, 1], [[-1], [0], [1]], [0, 0, 1], learning_rate=0.2, epochs=1)
        result = train_logistic_classifier(data)
        np.testing.assert_allclose(result.weights, [0.1])
        self.assertAlmostEqual(result.bias, 0)
        np.testing.assert_allclose(result.probabilities, [1/(1+np.exp(0.1)), 0.5, 1/(1+np.exp(-0.1))])
        np.testing.assert_array_equal(result.predictions, [0, 1, 1])
        self.assertAlmostEqual(result.loss_history[0], float(np.log1p(np.exp(-0.1))))

    def test_logistic_separable_data_and_extreme_logits(self):
        data = ClassificationInput([[-2], [-1], [1], [2]], [0, 0, 1, 1], [[-1e6], [1e6]], [0, 1])
        result = train_logistic_classifier(data)
        self.assertEqual(len(result.loss_history), data.epochs)
        self.assertTrue(np.isfinite(result.loss_history).all())
        self.assertLess(result.loss_history[-1], result.loss_history[0])
        np.testing.assert_allclose(result.probabilities, [0, 1])
        np.testing.assert_array_equal(result.predictions, [0, 1])

    def test_constant_class_is_supported_by_own_model(self):
        result = train_logistic_classifier(ClassificationInput([[0], [0]], [1, 1], [[0]], [1]))
        self.assertGreater(result.probabilities[0], 0.98)

    def test_test_labels_do_not_affect_training(self):
        for cls, train in [(RegressionInput, train_linear_regression), (ClassificationInput, train_logistic_classifier)]:
            with self.subTest(model=cls.__name__):
                data = cls([[-1], [1]], [0, 1], [[0]], [0])
                a, b = train(data), train(replace(data, test_target=[999]))
                np.testing.assert_array_equal(a.weights, b.weights)
                self.assertEqual(a.bias, b.bias)
                self.assertEqual(a.loss_history, b.loss_history)

    def test_inputs_are_not_modified(self):
        x = np.array([[-1.0], [1.0]])
        y = np.array([0.0, 1.0])
        for cls, train in [(RegressionInput, train_linear_regression), (ClassificationInput, train_logistic_classifier)]:
            train(cls(x, y, x, y))
        np.testing.assert_array_equal(x, [[-1], [1]])
        np.testing.assert_array_equal(y, [0, 1])

    def test_invalid_shapes_and_hyperparameters(self):
        base = RegressionInput([[0], [1]], [0, 1], [[0]], [0])
        for changes in ({"train_target": [0]}, {"test_features": [[0, 1]]}, {"epochs": 0}, {"learning_rate": -1}, {"train_features": [[np.nan], [1]]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                train_linear_regression(replace(base, **changes))

    def test_invalid_binary_labels(self):
        with self.assertRaises(ValueError):
            train_logistic_classifier(ClassificationInput([[0], [1]], [0, 2], [[0]], [0]))

    def test_baselines_on_small_sample(self):
        result = compare_classical_models(ClassificationInput([[-2], [-1], [1], [2]], [0, 0, 1, 1], [[-1.5], [1.5]], [0, 1]))
        for metrics in asdict(result).values():
            for value in metrics.values():
                self.assertTrue(0 <= value <= 1)
        self.assertEqual(result.decision_tree.accuracy, 1)

    def test_auc_requires_both_test_classes(self):
        with self.assertRaisesRegex(ValueError, "ROC-AUC"):
            classification_metrics([1, 1], [1, 1], [0.7, 0.8])


class DatasetTests(unittest.TestCase):
    def test_scaling_is_fitted_on_train(self):
        regression, classification = prepare_data()
        for data in (regression, classification):
            np.testing.assert_allclose(data.train_features.mean(axis=0), 0, atol=1e-12)
            self.assertFalse(np.allclose(data.test_features.mean(axis=0), 0, atol=1e-12))
        self.assertEqual((len(regression.train_target), len(regression.test_target)), (8000, 2000))
        self.assertEqual((len(classification.train_target), len(classification.test_target)), (800, 200))
        np.testing.assert_array_equal(np.bincount(classification.test_target.astype(int)), [60, 140])

    def test_full_experiment_reproduces_saved_metrics_and_renders(self):
        root = Path(__file__).resolve().parent
        expected = json.loads((root / "experiment_results.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            actual = run(directory, show=False)
            self.assertAlmostEqual(actual["linear_regression"]["test_mse"], expected["linear_regression"]["test_mse"], delta=1e-6)
            for model, metrics in expected["classification"].items():
                for metric, value in metrics.items():
                    self.assertAlmostEqual(actual["classification"][model][metric], value, delta=1e-6)
            for name in ("learning_curves.png", "roc_curves.png"):
                self.assertGreater((Path(directory) / name).stat().st_size, 1000)

    def test_visual_output(self):
        import matplotlib.pyplot as plt
        root = Path(__file__).resolve().parent
        for name, title in [("learning_curves.png", "Training losses"), ("roc_curves.png", "ROC curves")]:
            image = plt.imread(root / name)
            self.assertEqual(image.ndim, 3)
            figure, axis = plt.subplots(figsize=(10, 5))
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
            figure.tight_layout()
        if "agg" != plt.get_backend().lower():
            plt.show()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
