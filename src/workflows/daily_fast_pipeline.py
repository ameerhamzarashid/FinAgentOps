from src.workflows.daily_pipeline import daily_finagentops_pipeline


if __name__ == "__main__":
    daily_finagentops_pipeline(
        retrain_models=False,
        run_predictions=True,
        run_agent_reports=True,
    )