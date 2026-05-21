import argparse
import json
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO


def evaluate(weights: str, dataset: str, cfg: dict) -> dict:
    Path("metrics").mkdir(exist_ok=True)
    model = YOLO(weights)
    results = model.val(data=dataset)

    metrics = {
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "fire_mAP50": 0.0,
        "smoke_mAP50": 0.0,
    }
    with open("metrics/eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="params.yaml")
    p.add_argument("--weights", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--check-threshold", action="store_true")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset = "data/sample/dataset.yaml" if args.smoke else "data/dataset.yaml"
    metrics = evaluate(args.weights, dataset, cfg)
    if args.check_threshold and not args.smoke:
        thresh = cfg["evaluate"]["smoke_map_threshold"]
        if metrics["smoke_mAP50"] < thresh:
            print(f"FAIL: smoke_mAP50 {metrics['smoke_mAP50']:.3f} < {thresh}")
            sys.exit(1)
