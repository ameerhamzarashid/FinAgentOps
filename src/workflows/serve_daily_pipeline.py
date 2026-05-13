from src.workflows.daily_pipeline import daily_finagentops_pipeline


if __name__ == "__main__":
    daily_finagentops_pipeline.serve(
        name="daily-finagentops-fast-run",
        cron="0 18 * * 1-5",
        parameters={
            "retrain_models": False,
            "run_predictions": True,
        },
    )