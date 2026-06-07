import json
import mlflow
import dagshub
from mlflow.tracking import MlflowClient

MODEL_NAME = "firewatch-detector"
REPO_OWNER = "islam_tb"
REPO_NAME = "firewatch-ci"


def register_latest_run():
    with open("metrics/train_results.json") as f:
        results = json.load(f)
    run_id = results.get("run_id")
    if not run_id:
        raise ValueError("No run_id in metrics/train_results.json! Run train.py first.")

    print(f"Connecting to DagsHub MLflow server for run: {run_id}...")
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment(REPO_NAME)

    client = MlflowClient()

    print(f"Registering model version under name '{MODEL_NAME}'...")
    mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)

    client.set_model_version_tag(MODEL_NAME, mv.version, "stage", "staging")
    client.set_model_version_tag(
        MODEL_NAME, mv.version, "mAP50", str(results.get("mAP50", "unknown"))
    )

    print(f"Success! Registered {MODEL_NAME} v{mv.version} from remote run {run_id}")
    print(f"mAP50: {results.get('mAP50', 'N/A')}")


if __name__ == "__main__":
    register_latest_run()
