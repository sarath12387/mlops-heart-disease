import os

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed; render plots to files
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor

DATA_PATH = "data/processed/heart_cleaned.csv"
MODEL_PATH = "models/model.pkl"
SCORING = ["accuracy", "precision", "recall", "roc_auc"]
EXPERIMENT_NAME = "heart-disease-classification"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def build_models():
    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", build_preprocessor()),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]),
    }


def evaluate_models(models, X, y):
    results = {}
    for name, pipe in models.items():
        with mlflow.start_run(run_name=f"cv_{name}"):
            cv = cross_validate(pipe, X, y, cv=5, scoring=SCORING)
            mlflow.log_param("model_type", name)
            mlflow.log_param("cv_folds", 5)

            results[name] = {}
            print(f"\n=== {name} ===")
            for m in SCORING:
                mean_score = cv[f"test_{m}"].mean()
                results[name][m] = mean_score
                mlflow.log_metric(f"cv_{m}_mean", mean_score)
                mlflow.log_metric(f"cv_{m}_std", cv[f"test_{m}"].std())
                print(f"{m}: {mean_score:.3f} (+/- {cv[f'test_{m}'].std():.3f})")
    return results


def tune_best_model(models, X, y):
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
    }
    grid = GridSearchCV(
        models["logistic_regression"],
        param_grid,
        cv=5,
        scoring="roc_auc",
    )
    grid.fit(X, y)
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best ROC-AUC: {grid.best_score_:.3f}")
    return grid


def log_evaluation_plots(pipe, X, y):
    """Hold-out plots for the final model, logged as MLflow artifacts."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_tr, y_tr)

    os.makedirs("plots", exist_ok=True)

    RocCurveDisplay.from_estimator(pipe, X_te, y_te)
    plt.title("ROC Curve - Final Model (hold-out)")
    plt.savefig("plots/roc_curve.png", bbox_inches="tight")
    plt.close()

    ConfusionMatrixDisplay.from_estimator(pipe, X_te, y_te)
    plt.title("Confusion Matrix - Final Model (hold-out)")
    plt.savefig("plots/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    mlflow.log_artifact("plots/roc_curve.png")
    mlflow.log_artifact("plots/confusion_matrix.png")


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    models = build_models()
    evaluate_models(models, X, y)

    with mlflow.start_run(run_name="final_tuned_model"):
        grid = tune_best_model(models, X, y)
        best_pipe = grid.best_estimator_

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_roc_auc", grid.best_score_)

        log_evaluation_plots(best_pipe, X, y)

        best_pipe.fit(X, y)  # final fit on all data before shipping
        mlflow.sklearn.log_model(best_pipe, name="model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_pipe, MODEL_PATH)
        print(f"\nSaved final pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()