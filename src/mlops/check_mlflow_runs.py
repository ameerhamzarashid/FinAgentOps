import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "stock_direction_baseline_models"


def check_mlflow_runs():
    mlflow.set_tracking_uri("file:./mlruns")

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print(f"No experiment found with name: {EXPERIMENT_NAME}")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
    )

    print(f"\nExperiment: {EXPERIMENT_NAME}")
    print(f"Total runs: {len(runs)}\n")

    for run in runs:
        print("-" * 50)
        print(f"Run name: {run.data.tags.get('mlflow.runName')}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {run.data.metrics.get('accuracy')}")
        print(f"Precision: {run.data.metrics.get('precision')}")
        print(f"Recall: {run.data.metrics.get('recall')}")
        print(f"F1 Score: {run.data.metrics.get('f1_score')}")


if __name__ == "__main__":
    check_mlflow_runs()