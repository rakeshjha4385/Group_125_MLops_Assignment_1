# Databricks notebook source
# MAGIC %md
# MAGIC ## Reusable MLflow Logging Function

# COMMAND ----------

# Get current git commit hash
def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        return "unknown"

# COMMAND ----------

# function to log parameters, metrics, figures, and artifacts to MLflow
def log_mlflow_run(
    model, 
    params: dict, 
    metrics: dict = None, 
    figures: dict = None, 
    artifacts: dict = None, 
    model_name: str = None,
    run_name: str = None
):
    """
    A standard function for logging ML experiments to MLflow.
    Any DS can call this for consistent tracking.

    Args:
        model: model object (KShape, Prophet, etc.)
        params: dictionary of hyperparameters
        metrics: dictionary of evaluation metrics
        figures: dict of {figure_name: matplotlib_figure}
        artifacts: dict of {filename: local_path}
        model_name: for registry (optional)
        run_name: name of MLflow run
    """
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("git_branch", "main")
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

        # Log plots
        if figures:
            for name, fig in figures.items():
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                mlflow.log_figure(fig, f"{name}.png")
                plt.close(fig)

        # Log artifacts
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)

        # Log model
        if model:
            if model_name:
                mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
            else:
                mlflow.sklearn.log_model(model, "model")


# COMMAND ----------

