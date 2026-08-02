from pathlib import Path
import argparse
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "train.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "artifacts" / "test.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
TARGET = "ProdTaken"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = Path(args.train_file)
    test_path = Path(args.test_file)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Train/test files not found: {train_path}, {test_path}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column for column in X_train.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    parameter_distributions = {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [None, 8, 12, 16, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=12,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment("wellness_package_prediction")

    with mlflow.start_run(run_name="random_forest_tuning") as parent_run:
        search.fit(X_train, y_train)

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results_path = REPORT_DIR / "cv_results.csv"
        cv_results.to_csv(cv_results_path, index=False)

        # Log every tested parameter combination as a nested MLflow run.
        for index, row in cv_results.iterrows():
            with mlflow.start_run(
                run_name=f"candidate_{index + 1}",
                nested=True
            ):
                candidate_params = row["params"]
                mlflow.log_params(candidate_params)
                mlflow.log_metric(
                    "mean_cv_f1",
                    float(row["mean_test_score"])
                )
                mlflow.log_metric(
                    "std_cv_f1",
                    float(row["std_test_score"])
                )
                mlflow.log_metric(
                    "mean_train_f1",
                    float(row["mean_train_score"])
                )

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "best_cv_f1": float(search.best_score_)
        }

        report = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )

        model_path = MODEL_DIR / "wellness_package_model.joblib"
        metadata_path = MODEL_DIR / "model_metadata.json"
        metrics_path = REPORT_DIR / "metrics.json"
        report_path = REPORT_DIR / "classification_report.json"
        confusion_path = REPORT_DIR / "confusion_matrix.png"

        joblib.dump(best_model, model_path)

        metadata = {
            "target": TARGET,
            "model_type": "RandomForestClassifier",
            "best_parameters": search.best_params_,
            "numeric_features": numeric_columns,
            "categorical_features": categorical_columns,
            "training_rows": int(len(X_train)),
            "testing_rows": int(len(X_test))
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=4), encoding="utf-8"
        )
        metrics_path.write_text(
            json.dumps(metrics, indent=4), encoding="utf-8"
        )
        report_path.write_text(
            json.dumps(report, indent=4), encoding="utf-8"
        )

        cm = confusion_matrix(y_test, predictions)
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Purchased", "Purchased"]
        ).plot()
        plt.title("Random Forest Confusion Matrix")
        plt.tight_layout()
        plt.savefig(confusion_path, dpi=150)
        plt.close()

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cv_results_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(confusion_path))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
        )

    print("=" * 65)
    print("MODEL TRAINING SUMMARY")
    print("=" * 65)
    print("Best parameters:")
    print(json.dumps(search.best_params_, indent=4))
    print("\\nTest metrics:")
    print(json.dumps(metrics, indent=4))
    print(f"\\nSaved model: {model_path}")
    print(f"MLflow run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
