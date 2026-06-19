import json
import os
import time

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

    try:
        client.get_registered_model(MODEL_NAME)
    except Exception:
        print(f"Model {MODEL_NAME} not found in registry. Creating it.")
        client.create_registered_model(MODEL_NAME)

    # Register new version from the training run
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)

    # Wait until the version is ready before tagging and aliasing it
    for _ in range(30):
        current = client.get_model_version(MODEL_NAME, mv.version)
        if current.status == "READY":
            break
        time.sleep(2)

    # Tag with evaluation metrics for full traceability in the Registry UI
    for key, val in eval_metrics.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, key, str(val))

    # Point the production alias to the newly registered version
    client.set_registered_model_alias(MODEL_NAME, "production", mv.version)

    print(f"Registered   {MODEL_NAME} v{mv.version}  run={run_id}")
    print("Promoted to  production")
    print(f"  fire_mAP50:  {eval_metrics.get('fire_mAP50',  'N/A')}")
    print(f"  smoke_mAP50: {eval_metrics.get('smoke_mAP50', 'N/A')}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    register_and_promote()
