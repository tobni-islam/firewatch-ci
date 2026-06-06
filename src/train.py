import argparse
import json
import os
import random
from pathlib import Path
import mlflow
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# Try importing dagshub for remote logging integration
try:
    import dagshub
except ImportError:
    dagshub = None


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(cfg, epochs, smoke=False):
    set_seeds(cfg["train"]["seed"])
    dataset = "data/sample/dataset.yaml" if smoke else "data/dataset.yaml"

    out_dir_base = "models/smoke_weights" if smoke else "models/weights"
    out_dir_abs = Path(out_dir_base).resolve()

    mfile = "metrics/smoke_results.json" if smoke else "metrics/train_results.json"
    bs = cfg["smoke_train" if smoke else "train"]["batch_size"]

    out_dir_abs.mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    if dagshub and "DAGSHUB_TOKEN" in os.environ:
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "islam-tb")
        repo_name = os.getenv("DAGSHUB_REPO_NAME", "firewatch-ci")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    else:
        print("DagsHub credentials missing or library not installed. Logging locally.")

    mlflow.set_experiment("firewatch-ci")
    with mlflow.start_run(run_name="smoke-train" if smoke else "full-train") as run:
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": bs,
                "img_size": cfg["model"]["img_size"],
                "architecture": cfg["model"]["architecture"],
                "smoke_run": smoke,
                "lr0": cfg["train"]["lr0"],
            }
        )
        model = YOLO(f"{cfg['model']['architecture']}.pt")

        aug = {
            k: cfg["train"].get(k, v)
            for k, v in {
                "mosaic": 1.0,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "fliplr": 0.5,
                "degrees": 0.0,
                "scale": 0.5,
                "translate": 0.1,
                "cls": 1.5,
            }.items()
        }

        results = model.train(
            data=dataset,
            epochs=epochs,
            imgsz=cfg["model"]["img_size"],
            batch=bs,
            lr0=cfg["train"]["lr0"],
            weight_decay=cfg["train"]["weight_decay"],
            project=str(out_dir_abs),
            name="train",
            exist_ok=True,
            seed=cfg["train"]["seed"],
            save_period=10,
            **aug,
        )

        rd = results.results_dict
        metrics = {
            "run_id": run.info.run_id,
            "mAP50": round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
            "precision": round(float(rd.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(rd.get("metrics/recall(B)", 0.0)), 4),
        }

        with open(mfile, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"run_id : {run.info.run_id}")
        print(f"mAP50  : {metrics['mAP50']:.4f}")

        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "run_id"})

        best_weights = out_dir_abs / "train" / "weights" / "best.pt"
        if best_weights.exists():
            try:
                mlflow.log_artifact(str(best_weights), artifact_path="model")
                print("Successfully logged weights artifact to MLflow.")
            except Exception as e:
                print(f"Failed to log artifact to MLflow: {e}")
        else:
            print(f"Expected weights file not found at: {best_weights}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="params.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    epochs = args.epochs or (
        cfg["smoke_train"]["epochs"] if args.smoke else cfg["train"]["epochs"]
    )
    train(cfg, epochs=epochs, smoke=args.smoke)
