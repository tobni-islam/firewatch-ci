"""src/train.py — Week 2: full MLflow logging."""

import argparse
import json
import random
from pathlib import Path
import mlflow
import numpy as np
import torch
import yaml
from ultralytics import YOLO


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(cfg, epochs, smoke=False):
    set_seeds(cfg["train"]["seed"])
    dataset = "data/sample/dataset.yaml" if smoke else "data/dataset.yaml"
    out_dir = "models/smoke_weights" if smoke else "models/weights"
    mfile = "metrics/smoke_results.json" if smoke else "metrics/train_results.json"
    bs = cfg["smoke_train" if smoke else "train"]["batch_size"]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

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
        results = model.train(
            data=dataset,
            epochs=epochs,
            imgsz=cfg["model"]["img_size"],
            batch=bs,
            lr0=cfg["train"]["lr0"],
            weight_decay=cfg["train"]["weight_decay"],
            project=out_dir,
            name="train",
            exist_ok=True,
            seed=cfg["train"]["seed"],
            save_period=10,  # checkpoint every 10 epochs — critical for Colab free tier
        )
        rd = results.results_dict
        metrics = {
            "run_id": run.info.run_id,
            "mAP50": round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
            "precision": round(float(rd.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(rd.get("metrics/recall(B)", 0.0)), 4),
        }
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "run_id"})
        mlflow.log_artifact(f"{out_dir}/train/weights/best.pt", artifact_path="model")
        with open(mfile, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"run_id : {run.info.run_id}")
        print(f"mAP50  : {metrics['mAP50']:.4f}")


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
