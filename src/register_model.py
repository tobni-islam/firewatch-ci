import json
import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "firewatch-detector"


def register_and_promote() -> None:
    client = MlflowClient()

    with open("metrics/train_results.json") as f:
        train_meta = json.load(f)
    run_id = train_meta["run_id"]

    with open("metrics/eval_results.json") as f:
        eval_metrics = json.load(f)

    # Archive current Production version before promoting new one
    for v in client.get_latest_versions(MODEL_NAME, stages=["Production"]):
        client.transition_model_version_stage(MODEL_NAME, v.version, "Archived")
        print(f"Archived previous Production v{v.version}")

    # Register new version from the Colab/Kaggle training run
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)

    # Tag with evaluation metrics for full traceability in the Registry UI
    for key, val in eval_metrics.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, key, str(val))

    client.transition_model_version_stage(MODEL_NAME, mv.version, "Production")

    print(f"Registered   {MODEL_NAME} v{mv.version}  run={run_id}")
    print("Promoted to  Production")
    print(f"  fire_mAP50:  {eval_metrics.get('fire_mAP50',  'N/A')}")
    print(f"  smoke_mAP50: {eval_metrics.get('smoke_mAP50', 'N/A')}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    register_and_promote()
