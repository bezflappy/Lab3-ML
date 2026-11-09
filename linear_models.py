import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    ClassificationMetrics,
    RegressionInput,
    RegressionResult,
)


def _inputs(data):
    x = np.asarray(data.train_features, dtype=float)
    y = np.asarray(data.train_target, dtype=float)
    test = np.asarray(data.test_features, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if x.ndim != 2 or test.ndim != 2 or y.ndim != 1:
        raise ValueError("Features must be matrices and target a vector")
    if not len(x) or not x.shape[1] or len(x) != len(y) or x.shape[1] != test.shape[1]:
        raise ValueError("Empty training data or incompatible dimensions")
    if not all(np.isfinite(a).all() for a in (x, y, test)):
        raise ValueError("Inputs must contain only finite numbers")
    if not np.isfinite(data.learning_rate) or data.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(data.epochs, (int, np.integer)) or data.epochs < 1:
        raise ValueError("epochs must be a positive integer")
    return x, y, test


def _sigmoid(z):
    exponential = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + exponential), exponential / (1 + exponential))


def classification_metrics(target, predictions, probabilities):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    target = np.asarray(target).reshape(-1)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("ROC-AUC requires both binary classes in test_target")
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def baseline_models(train_size):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=min(5, train_size)),
    }


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    x, y, test = _inputs(data)
    weights = np.zeros(x.shape[1])
    bias = 0.0
    history = []
    for _ in range(data.epochs):
        error = x @ weights + bias - y
        weights -= data.learning_rate * (2 / len(y)) * (x.T @ error)
        bias -= data.learning_rate * 2 * float(error.mean())
        loss = float(np.mean((x @ weights + bias - y) ** 2))
        if not np.isfinite(loss):
            raise ValueError("Training diverged: scale features or reduce learning_rate")
        history.append(loss)
    return RegressionResult(weights, float(bias), history, test @ weights + bias)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    x, y, test = _inputs(data)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary labels must be 0 or 1")
    weights = np.zeros(x.shape[1])
    bias = 0.0
    history = []
    for _ in range(data.epochs):
        error = _sigmoid(x @ weights + bias) - y
        weights -= data.learning_rate * (x.T @ error) / len(y)
        bias -= data.learning_rate * float(error.mean())
        logits = x @ weights + bias
        loss = float(np.mean(np.logaddexp(0, logits) - y * logits))
        if not np.isfinite(loss):
            raise ValueError("Training diverged: scale features or reduce learning_rate")
        history.append(loss)
    probabilities = _sigmoid(test @ weights + bias)
    return ClassificationResult(
        weights, float(bias), history, probabilities, (probabilities >= 0.5).astype(int)
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    x, y, test = _inputs(data)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Baseline training requires both binary classes")
    metrics = {}
    for name, model in baseline_models(len(y)).items():
        model.fit(x, y)
        probabilities = model.predict_proba(test)[:, list(model.classes_).index(1)]
        metrics[name] = classification_metrics(data.test_target, model.predict(test), probabilities)
    return BaselineComparisonResult(**metrics)
