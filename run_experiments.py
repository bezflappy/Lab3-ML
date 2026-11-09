"""Run deterministic experiments, save metrics, report and figures; --show opens plots."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    baseline_models, classification_metrics, compare_classical_models,
    train_linear_regression, train_logistic_classifier,
)

ROOT = Path(__file__).resolve().parent


def prepare_data():
    students = pd.read_csv(ROOT / "Student_Performance.csv")
    credit = pd.read_csv(ROOT / "german.csv", sep=";")
    if students.isna().any().any() or credit.isna().any().any():
        raise ValueError("Unexpected missing values in supplied datasets")
    student_x = students.drop(columns="Performance Index").copy()
    student_x["Extracurricular Activities"] = student_x["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    if student_x.isna().any().any():
        raise ValueError("Unknown extracurricular activity category")
    sx, st, sy, sty = train_test_split(
        student_x, students["Performance Index"], test_size=0.2, random_state=42
    )
    cx, ct, cy, cty = train_test_split(
        credit.iloc[:, 1:], credit.iloc[:, 0], test_size=0.2, random_state=42,
        stratify=credit.iloc[:, 0],
    )
    ss, cs = StandardScaler(), StandardScaler()
    regression = RegressionInput(ss.fit_transform(sx), sy.to_numpy(), ss.transform(st), sty.to_numpy())
    classification = ClassificationInput(cs.fit_transform(cx), cy.to_numpy(), cs.transform(ct), cty.to_numpy())
    return regression, classification


def run(output_dir=ROOT, show=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regression, classification = prepare_data()
    linear = train_linear_regression(regression)
    own = train_logistic_classifier(classification)
    comparisons = compare_classical_models(classification)
    results = {
        "linear_regression": {"test_mse": float(mean_squared_error(regression.test_target, linear.predictions))},
        "classification": {
            "own_logistic": asdict(classification_metrics(classification.test_target, own.predictions, own.probabilities)),
            **asdict(comparisons),
        },
    }
    experiments = []
    for rate in (0.01, 0.05, 0.1):
        result = train_linear_regression(replace(regression, learning_rate=rate))
        experiments.append({"model": "linear", "learning_rate": rate, "epochs": regression.epochs,
                            "train_loss_first": result.loss_history[0], "train_loss_last": result.loss_history[-1],
                            "test_mse": float(mean_squared_error(regression.test_target, result.predictions))})
    for rate in (0.03, 0.1):
        for epochs in (300, 1000):
            result = train_logistic_classifier(replace(classification, learning_rate=rate, epochs=epochs))
            experiments.append({"model": "own_logistic", "learning_rate": rate, "epochs": epochs,
                                "train_loss_first": result.loss_history[0], "train_loss_last": result.loss_history[-1],
                                **asdict(classification_metrics(classification.test_target, result.predictions, result.probabilities))})
    (output_dir / "experiment_results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output_dir / "experiment_details.json").write_text(json.dumps(experiments, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    import matplotlib
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(np.arange(1, regression.epochs+1), linear.loss_history)
    axes[0].set(xlabel="Epoch", ylabel="Train MSE", title="Own linear regression", yscale="log")
    axes[1].plot(np.arange(1, classification.epochs+1), own.loss_history)
    axes[1].set(xlabel="Epoch", ylabel="Train BCE", title="Own logistic classifier")
    figure.savefig(output_dir / "learning_curves.png", dpi=160)
    roc_figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    probabilities = {"own_logistic": own.probabilities}
    for name, model in baseline_models(len(classification.train_target)).items():
        model.fit(classification.train_features, classification.train_target)
        probabilities[name] = model.predict_proba(classification.test_features)[:, 1]
    for name, values in probabilities.items():
        fpr, tpr, _ = roc_curve(classification.test_target, values)
        axis.plot(fpr, tpr, label=f"{name}: AUC={results['classification'][name]['roc_auc']:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="German credit: same test split")
    axis.legend()
    roc_figure.savefig(output_dir / "roc_curves.png", dpi=160)

    metric_rows = "\n".join("| " + name + " | " + " | ".join(f"{m[k]:.6f}" for k in ("roc_auc","accuracy","precision","recall","f1")) + " |" for name, m in results["classification"].items())
    experiment_rows = "\n".join(f"| {e['model']} | {e['learning_rate']} | {e['epochs']} | {e['train_loss_first']:.6f} | {e['train_loss_last']:.6f} | {e.get('test_mse',e.get('roc_auc')):.6f} |" for e in experiments)
    own_metrics = results["classification"]["own_logistic"]
    differences = "\n".join(f"- Против {name}: разница ROC-AUC {own_metrics['roc_auc']-m['roc_auc']:+.6f}; Accuracy {own_metrics['accuracy']-m['accuracy']:+.6f}." for name,m in results["classification"].items() if name!="own_logistic")
    report = f'''# Лабораторная №3. Линейные модели

## 1. Данные и подготовка

Student_Performance.csv: {len(regression.train_target)+len(regression.test_target)} наблюдений, {regression.train_features.shape[1]} признаков; цель Performance Index. Extracurricular Activities: Yes → 1, No → 0. Разбиение 80/20, train={len(regression.train_target)}, test={len(regression.test_target)}, random_state=42.

german.csv читается с разделителем `;`: {len(classification.train_target)+len(classification.test_target)} наблюдений, {classification.train_features.shape[1]} признаков; первый столбец Creditability — цель. Стратифицированное разбиение 80/20: train={len(classification.train_target)}, test={len(classification.test_target)}. В test: класс 0 — {int(np.sum(classification.test_target==0))}, класс 1 — {int(np.sum(classification.test_target==1))}. Положительный класс для Precision/Recall/F1 — 1 (Creditability=1); это не метрики поиска класса 0.

Пропусков в обоих файлах нет (проверено кодом). Числовые коды категорий German используются как предоставлены: это простой базовый эксперимент; расстояния между номинальными кодами не имеют гарантированного содержательного смысла, особенно для KNN. One-hot кодирование — направление дальнейшего улучшения, в эти результаты не включено.

Для каждого датасета отдельный StandardScaler обучается **только на train**; test преобразуется теми же средними и масштабами. Цель регрессии не масштабируется. Веса функций относятся к переданным, в эксперименте стандартизованным, признакам. Все классификаторы используют одинаковые train/test и одинаковые признаки; тестовые цели не участвуют в обучении.

## 2. Гипотезы и эксперименты

Своя регрессия: нулевые веса и смещение, полный пакет, MSE; градиенты `2 X.T @ error / n` и `2 mean(error)`. Основной запуск: learning_rate=0.05, epochs=800. Свой классификатор: линейный logit, устойчивая sigmoid, BCE через logaddexp; градиенты `X.T @ (p-y) / n` и `mean(p-y)`. Основной запуск: learning_rate=0.1, epochs=1000, порог 0.5. История содержит loss после каждого обновления, её длина равна epochs. Регуляризации и ранней остановки в собственных моделях нет.

Гипотезы: при масштабировании MSE устойчиво снижается; меньший шаг требует большего числа эпох; дерево и KNN могут учитывать нелинейные зависимости, но чувствительны к переобучению и кодированию признаков. Ниже реальные запуски с заранее фиксированными параметрами. Основные настройки взяты из контрактов и не выбирались по лучшей test-метрике. Для настоящего подбора потребуется отдельная validation/CV.

| Модель | Шаг | Эпохи | Первый train loss | Последний train loss | Test MSE / ROC-AUC |
|---|---:|---:|---:|---:|---:|
{experiment_rows}

Готовые модели: sklearn LogisticRegression, C=1, penalty по умолчанию (L2), max_iter=2000; DecisionTreeClassifier без ограничения глубины, random_state=42; KNeighborsClassifier, k=5, евклидово расстояние и равные веса. Для обучающей выборки менее пяти строк k ограничивается её размером.

## 3. Метрики

Линейная регрессия: test MSE = **{results['linear_regression']['test_mse']:.6f}**. Train MSE после первой эпохи = {linear.loss_history[0]:.6f}, после последней = {linear.loss_history[-1]:.6f}.

Свой классификатор: train BCE {own.loss_history[0]:.6f} → {own.loss_history[-1]:.6f}. ROC-AUC считается по вероятностям, остальные метрики — по меткам. Деление на ноль в Precision/Recall/F1 обрабатывается `zero_division=0`; ROC-AUC требует оба класса на test.

| Модель | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
{metric_rows}

![Ход обучения](learning_curves.png)

![ROC-кривые](roc_curves.png)

## 4. Выводы

Все три шага линейной регрессии к 800-й эпохе дали практически одинаковую test MSE; большой шаг быстрее снижает начальную ошибку. Для логистической модели снижение train BCE не гарантирует улучшения test ROC-AUC: при шаге 0.03 и 300 эпохах AUC = {experiments[3]['roc_auc']:.6f}, а в основном запуске = {own_metrics['roc_auc']:.6f}. Это аргумент в пользу проверки ранней остановки на отдельной validation-выборке, а не выбора лучшего запуска по test.

Собственные модели обучены пакетным градиентным спуском без готовых линейных моделей sklearn внутри двух train-функций. Снижение тренировочной ошибки видно по таблице и графикам. Сравнение собственной логистической модели с остальными (собственная минус baseline):

{differences}

Различие со sklearn ожидаемо: у эталона L2-регуляризация и другой оптимизатор, у нашей модели фиксированное число шагов без регуляризации. Дерево без ограничения глубины может переобучаться; KNN зависит от масштаба и смысла расстояний. Это объяснения возможных причин, а не доказательство причинности по одному разбиению. Результаты относятся к seed=42 и не доказывают превосходство на других выборках; для устойчивого вывода нужна кросс-валидация.

В README упомянуты собственная логистическая модель, дерево и KNN, а docstring compare_classical_models и схема JSON требуют также готовую LogisticRegression. Поэтому сравниваются все четыре: функция compare_classical_models заполняет три поля контракта готовыми baseline-моделями, собственная модель рассчитана отдельно на **том же** split и включена в REPORT и JSON как own_logistic.

## 5. Воспроизводимость

Запуск из папки проекта (Python 3.12+ для зафиксированных версий библиотек):

```bash
python -m pip install -r requirements.txt
python run_experiments.py
python -m unittest discover -v
```

Для просмотра графиков: `python run_experiments.py --show`. В обычной графической среде визуальный тест также показывает графики; закройте окна, чтобы завершить тесты. Для автоматического запуска без окон: `MPLBACKEND=Agg python -m unittest discover -v`.

Эксперимент пишет experiment_results.json, experiment_details.json, REPORT.md и два PNG в папку проекта. Параметры random_state=42; scaling обучается на train, классификационное разбиение стратифицировано. Фактическая среда: Python {platform.python_version()}, NumPy {np.__version__}, Pandas {pd.__version__}, scikit-learn {sklearn.__version__}, Matplotlib {matplotlib.__version__}.

JSON содержит числа полной точности. В предоставленном стартовом файле обещанная численная погрешность не указана; для собственной проверки выбран допуск 1e-6. Округление в отчёте — до шести знаков. Скрытые тесты преподавателя недоступны.
'''
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(results, indent=2))
    if show:
        plt.show()
    plt.close("all")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(show=args.show)
