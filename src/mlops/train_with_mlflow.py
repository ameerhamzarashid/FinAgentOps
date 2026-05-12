from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.database.connection import get_engine


EXPERIMENT_NAME = "stock_direction_baseline_models"

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/model/mlflow")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
    "daily_return",
    "log_return",
    "moving_avg_7",
    "moving_avg_14",
    "moving_avg_30",
    "volatility_7",
    "volatility_30",
    "rsi_14",
    "macd",
    "signal_line",
    "bollinger_upper",
    "bollinger_lower",
]


def load_training_data():
    engine = get_engine()

    query = """
    SELECT
        p.ticker,
        p.price_date,
        p.adjusted_close,
        f.daily_return,
        f.log_return,
        f.moving_avg_7,
        f.moving_avg_14,
        f.moving_avg_30,
        f.volatility_7,
        f.volatility_30,
        f.rsi_14,
        f.macd,
        f.signal_line,
        f.bollinger_upper,
        f.bollinger_lower
    FROM ticker_prices p
    INNER JOIN technical_features f
        ON p.ticker = f.ticker
        AND p.price_date = f.price_date
    ORDER BY p.ticker, p.price_date;
    """

    data = pd.read_sql(query, engine)
    data["price_date"] = pd.to_datetime(data["price_date"])

    return data


def create_target(data):
    data = data.copy()
    data = data.sort_values(["ticker", "price_date"])

    data["next_adjusted_close"] = (
        data.groupby("ticker")["adjusted_close"].shift(-1)
    )

    data["target"] = (
        data["next_adjusted_close"] > data["adjusted_close"]
    ).astype(int)

    return data


def prepare_dataset(data):
    data = create_target(data)

    required_columns = FEATURE_COLUMNS + ["target"]
    data = data.dropna(subset=required_columns)

    X = data[FEATURE_COLUMNS]
    y = data["target"]

    return X, y, data


def get_models():
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
    }

    return models


def save_confusion_matrix(model_name, y_test, predictions):
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["DOWN", "UP"],
    )

    display.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    plot_path = REPORT_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def train_and_log_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_rows,
):
    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name} with MLflow tracking...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_rows", dataset_rows)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("target", "next_day_direction")
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))

        if model_name == "random_forest":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("class_weight", model.class_weight)

        if model_name == "logistic_regression":
            logistic_model = model.named_steps["model"]
            mlflow.log_param("max_iter", logistic_model.max_iter)
            mlflow.log_param("scaler", "StandardScaler")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        report_text = classification_report(
            y_test,
            predictions,
            target_names=["DOWN", "UP"],
            zero_division=0,
        )

        report_path = REPORT_DIR / f"{model_name}_classification_report.txt"
        report_path.write_text(report_text)

        confusion_matrix_path = save_confusion_matrix(
            model_name=model_name,
            y_test=y_test,
            predictions=predictions,
        )

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(confusion_matrix_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        print(f"\nModel: {model_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }


def save_best_local_model(results, models):
    results_df = pd.DataFrame(results)

    results_path = REPORT_DIR / "mlflow_model_results.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.sort_values("f1_score", ascending=False).iloc[0]
    best_model_name = best_row["model_name"]
    best_model = models[best_model_name]

    model_path = MODEL_DIR / "mlflow_best_stock_direction_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model: {best_model_name}")
    print(f"Best local model saved to {model_path}")
    print(f"MLflow results saved to {results_path}")


def main():
    print("Starting Stage 4 MLflow MLOps tracking...")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_training_data()

    print(f"Loaded {len(data)} rows from database.")

    X, y, prepared_data = prepare_dataset(data)

    print(f"Prepared dataset with {len(X)} rows.")
    print("\nTarget distribution:")
    print(y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = get_models()

    results = []

    for model_name, model in models.items():
        result = train_and_log_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            dataset_rows=len(X),
        )
        results.append(result)

    save_best_local_model(results, models)

    print("\nStage 4 MLflow tracking completed successfully.")


if __name__ == "__main__":
    main()