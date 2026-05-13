from src.workflows.daily_pipeline import daily_finagentops_pipeline


if __name__ == "__main__":
    daily_finagentops_pipeline(
        retrain_models=True,
        run_predictions=True,
    )