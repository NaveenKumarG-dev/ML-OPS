from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.preprocess import create_preprocessor


DATA_PATH = Path("data/raw/bank-full.csv")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "bank-marketing"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def load_data() -> pd.DataFrame:
    """Load the Bank Marketing dataset."""

    return pd.read_csv(
        DATA_PATH,
        sep=";"
    )


def prepare_data(df: pd.DataFrame):
    """Separate features and target."""

    X = df.drop(columns=["y"])

    y = df["y"].map({
        "no": 0,
        "yes": 1
    })

    return X, y

def train_model(
    X_train,
    y_train,
    C=1.0,
    max_iter=1000,
    random_state=42,
):
    """Create preprocessing + model pipeline and train it."""

    preprocessor = create_preprocessor()

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline

def main():

    C = 1.0
    max_iter = 1000
    random_state = 42
    test_size = 0.2
    threshold = 0.30

    with mlflow.start_run():

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "C": C,
            "max_iter": max_iter,
            "random_state": random_state,
            "test_size": test_size,
            "threshold": threshold
        })

        mlflow.set_tags({
            "model_type": "logistic_regression",
            "dataset": "UCI Bank Marketing",
            "run_by" : "naveen"
        })

        print("Loading data...")

        df = load_data()

        print(f"Dataset shape: {df.shape}")


        X, y = prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")


        model = train_model(
            X_train,
            y_train,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )

        print("Model training completed.")

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test,)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        print(f"Predictions generated: {len(predictions)}")


        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities   
        )

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        })

        signature = infer_signature(
            model_input = X_train,
            model_output= model.predict(X_train)
        )

        mlflow.sklearn.log_model(
            sk_model = model,
            name="bank-marketing-model",
            signature = signature
        )
        print("\nEvaluation Results")
        print("------------------")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")


        run = mlflow.active_run()

        print("\nMLflow")
        print("------")
        print(f"Experiment : {MLFLOW_EXPERIMENT_NAME}")
        print(f"Run ID     : {run.info.run_id}")

if __name__ == "__main__":
    main()