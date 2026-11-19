
from collections import Counter

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "TrueLens Experiments"
)

if experiment is None:
    raise RuntimeError("TrueLens experiment not found.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=5000
)

print("\n========== TRUELENS MLFLOW AUDIT ==========")

print("Experiment:", experiment.name)
print("Experiment ID:", experiment.experiment_id)
print("Total recorded runs:", len(runs))

print("\n========== EXPERIMENT CATEGORIES ==========")

categories = Counter(
    run.data.tags.get("experiment_category", "uncategorized")
    for run in runs
)

for category, count in categories.items():
    print(f"{category}: {count} runs")

print("\n========== INDIVIDUAL RUNS ==========")

for run in runs:
    name = run.data.tags.get("mlflow.runName", "Unnamed")
    category = run.data.tags.get(
        "experiment_category", "uncategorized"
    )

    metric_count = len(run.data.metrics)
    param_count = len(run.data.params)

    try:
        artifacts = client.list_artifacts(
            run.info.run_id,
            "source_results"
        )
        artifact_count = len(artifacts)
    except Exception:
        artifact_count = 0

    print("\nRun:", name)
    print("Category:", category)
    print("Parameters:", param_count)
    print("Metrics:", metric_count)
    print("Source artifacts:", artifact_count)

print("\n========== AUDIT COMPLETED ==========")
