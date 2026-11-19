
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "TrueLens Experiments"
)

runs = client.search_runs(
    [experiment.experiment_id],
    max_results=5000
)

target_names = {
    "GenImage Expanded - EfficientNet-B0 Baseline",
    "GenImage Expanded - DINOv2 Baseline"
}

print("\n========== GENIMAGE DUPLICATE INSPECTION ==========")

for run in runs:

    name = run.data.tags.get("mlflow.runName", "")

    if name not in target_names:
        continue

    print("\nRun name:", name)
    print("Run ID:", run.info.run_id)
    print(
        "Created:",
        datetime.fromtimestamp(
            run.info.start_time / 1000
        ).isoformat()
    )
    print("Parameters:", run.data.params)
    print("Metrics:", run.data.metrics)
    print("Import key:", run.data.tags.get("historical_import_key"))

print("\nInspection completed. No data was modified.")
