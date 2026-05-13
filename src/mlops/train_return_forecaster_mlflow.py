from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.database.connection import get_engine


EXPERIMENT_NAME = "next_day_return_forecasting"

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/model/mlflow_forecasting")

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

    numeric_columns = ["adjusted_close"] + FEATURE_COLUMNS

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def create_next_day_return_target(data):
    data = data.copy()
    data = data.sort_values(["ticker", "price_date"])

    data["next_adjusted_close"] = (
        data.groupby("ticker")["adjusted_close"].shift(-1)
    )

    data["next_day_return"] = (
        data["next_adjusted_close"] - data["adjusted_close"]
    ) / data["adjusted_close"]

    return data


def prepare_dataset(data):
    data = create_next_day_return_target(data)

    required_columns = FEATURE_COLUMNS + ["next_day_return"]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required_columns)

    X = data[FEATURE_COLUMNS]
    y = data["next_day_return"]

    return X, y, data


def get_models():
    models = {
        "linear_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            min_samples_leaf=5,
        ),
    }

    return models


def save_actual_vs_predicted_plot(model_name, y_test, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.4)
    plt.xlabel("Actual Next-Day Return")
    plt.ylabel("Predicted Next-Day Return")
    plt.title(f"Actual vs Predicted Returns - {model_name}")
    plt.tight_layout()

    plot_path = REPORT_DIR / f"{model_name}_actual_vs_predicted.png"
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

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        actual_direction = (y_test > 0).astype(int)
        predicted_direction = (predictions > 0).astype(int)
        directional_accuracy = (actual_direction == predicted_direction).mean()

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", "next_day_return")
        mlflow.log_param("dataset_rows", dataset_rows)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))

        if model_name == "random_forest_regressor":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("min_samples_leaf", model.min_samples_leaf)

        if model_name == "ridge_regression":
            ridge_model = model.named_steps["model"]
            mlflow.log_param("alpha", ridge_model.alpha)
            mlflow.log_param("scaler", "StandardScaler")

        if model_name == "linear_regression":
            mlflow.log_param("scaler", "StandardScaler")

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("directional_accuracy", directional_accuracy)

        plot_path = save_actual_vs_predicted_plot(
            model_name=model_name,
            y_test=y_test,
            predictions=predictions,
        )

        result_text = (
            f"Model: {model_name}\n"
            f"MAE: {mae:.8f}\n"
            f"MSE: {mse:.8f}\n"
            f"RMSE: {rmse:.8f}\n"
            f"R2 Score: {r2:.8f}\n"
            f"Directional Accuracy: {directional_accuracy:.4f}\n"
        )

        report_path = REPORT_DIR / f"{model_name}_forecast_report.txt"
        report_path.write_text(result_text)

        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(report_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        print(result_text)

        return {
            "model_name": model_name,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "directional_accuracy": directional_accuracy,
        }


def save_best_local_model(results, models):
    results_df = pd.DataFrame(results)

    results_path = REPORT_DIR / "mlflow_return_forecasting_results.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.sort_values("rmse", ascending=True).iloc[0]
    best_model_name = best_row["model_name"]
    best_model = models[best_model_name]

    model_path = MODEL_DIR / "mlflow_next_day_return_forecaster.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model: {best_model_name}")
    print(f"Best local MLflow model saved to {model_path}")
    print(f"MLflow forecasting results saved to {results_path}")


def main():
    print("Starting Stage 5 MLflow return forecasting...")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_training_data()

    print(f"Loaded {len(data)} rows from database.")

    X, y, prepared_data = prepare_dataset(data)

    print(f"Prepared forecasting dataset with {len(X)} rows.")
    print("\nTarget summary:")
    print(y.describe())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
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

    print("\nStage 5 MLflow return forecasting completed successfully.")


if __name__ == "__main__":
    main()