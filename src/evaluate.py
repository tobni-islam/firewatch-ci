import argparse
import json
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO


def evaluate(weights, dataset, cfg, smoke=False):
    Path("metrics").mkdir(exist_ok=True)
    model = YOLO(weights)
    results = model.val(data=dataset, verbose=False)
    rd = results.results_dict

    # results.maps: per-class mAP50 array. class 0 = fire, class 1 = smoke.
    maps = list(results.maps) if hasattr(results, "maps") else []
    fire_map = round(float(maps[0]) if len(maps) > 0 else 0.0, 4)
    smoke_map = round(float(maps[1]) if len(maps) > 1 else 0.0, 4)

    metrics = {
        "mAP50": round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
        "fire_mAP50": fire_map,
        "smoke_mAP50": smoke_map,
    }
    out = "metrics/smoke_eval.json" if smoke else "metrics/eval_results.json"
    with open(out, "w") as f:
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
    metrics = evaluate(args.weights, dataset, cfg, smoke=args.smoke)

    if args.check_threshold and not args.smoke:
        failed = False
        if metrics["smoke_mAP50"] < cfg["evaluate"]["smoke_map_threshold"]:
            print(
                f"FAIL smoke_mAP50 {metrics['smoke_mAP50']:.3f} < {cfg['evaluate']['smoke_map_threshold']}"
            )
            failed = True
        if metrics["fire_mAP50"] < cfg["evaluate"]["fire_map_threshold"]:
            print(
                f"FAIL fire_mAP50 {metrics['fire_mAP50']:.3f} < {cfg['evaluate']['fire_map_threshold']}"
            )
            failed = True
        if failed:
            sys.exit(1)
