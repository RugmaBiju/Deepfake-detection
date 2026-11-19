
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

target_runs = {
    "GenImage Expanded - EfficientNet-B0 Baseline":
        "genimage_expanded_comparison_efficientnet_b0",

    "GenImage Expanded - DINOv2 Baseline":
        "genimage_expanded_comparison_dinov2",

    "GenImage Expanded - B0-DINOv2 Fusion":
        "genimage_expanded_comparison_b0_dinov2_fusion"
}

for name, import_key in target_runs.items():

    matching = [
        run for run in runs
        if run.data.tags.get("mlflow.runName") == name
    ]

    if not matching:
        raise RuntimeError(f"Missing run: {name}")

    matching.sort(
        key=lambda run: run.info.start_time,
        reverse=True
    )

    keeper = matching[0]

    # Verify every duplicate before deleting anything.
    for duplicate in matching[1:]:
        if (
            keeper.data.params != duplicate.data.params
            or keeper.data.metrics != duplicate.data.metrics
        ):
            raise RuntimeError(
                f"Results differ for {name}. Nothing deleted."
            )

    # Soft-delete older identical runs.
    for duplicate in matching[1:]:
        client.delete_run(duplicate.info.run_id)
        print("Soft-deleted duplicate:", duplicate.info.run_id)

    # Categorise the retained run.
    client.set_tag(
        keeper.info.run_id,
        "historical_import_key",
        import_key
    )

    client.set_tag(
        keeper.info.run_id,
        "run_origin",
        "historical_import"
    )

    client.set_tag(
        keeper.info.run_id,
        "dataset_track",
        "GenImage Expanded"
    )

    client.set_tag(
        keeper.info.run_id,
        "experiment_category",
        "image_detection"
    )

    client.set_tag(
        keeper.info.run_id,
        "evaluation_type",
        "test"
    )

    client.set_tag(
        keeper.info.run_id,
        "comparison_group",
        "GenImage Expanded Model Comparison"
    )

    print("Retained:", name)
    print("Run ID:", keeper.info.run_id)


active_runs = client.search_runs(
    [experiment.experiment_id],
    max_results=5000
)

print("\n========== CLEANUP SUMMARY ==========")
print("Active runs remaining:", len(active_runs))
print("Cleanup completed!")
