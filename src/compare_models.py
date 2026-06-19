import json
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "firewatch-detector"
MIN_IMPROVEMENT = 0.01


def compare() -> bool:
    client = MlflowClient()

    with open("metrics/eval_results.json") as f:
        new = json.load(f)
    new_map = new.get("mAP50", 0.0)
    new_smoke = new.get("smoke_mAP50", 0.0)
    new_fire = new.get("fire_mAP50", 0.0)

    try:
        prod = client.get_model_version_by_alias(MODEL_NAME, "production")
        prod_metrics = client.get_run(prod.run_id).data.metrics
        prod_map = prod_metrics.get("mAP50", 0.0)
        print(f"Baseline  mAP50={prod_map:.4f}")
    except Exception as exc:
        print(f"Warning: could not fetch baseline ({exc})! treating as first run.")
        prod_map = 0.0
        print("No baseline in registry! first promotion, auto-passing.")

    print(f"Candidate mAP50={new_map:.4f}  smoke={new_smoke:.4f}  fire={new_fire:.4f}")

    smoke_thresh = 0.2

    if prod_map == 0.0:
        return True

    passes = new_map > prod_map + MIN_IMPROVEMENT and new_smoke >= smoke_thresh
    if passes:
        print(f"PASS: +{new_map - prod_map:.4f} improvement over baseline.")
    else:
        reasons = []
        if new_map <= prod_map + MIN_IMPROVEMENT:
            reasons.append(
                f"mAP50 {new_map:.4f} not > {prod_map + MIN_IMPROVEMENT:.4f}"
            )
        if new_smoke < smoke_thresh:
            reasons.append(f"smoke_mAP50 {new_smoke:.4f} < threshold {smoke_thresh}")
        print(f"FAIL: {', '.join(reasons)}")
    return passes


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    if not compare():
        sys.exit(1)
