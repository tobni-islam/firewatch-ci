import argparse
import json
import random
from pathlib import Path
from turtle import pd

import mlflow
import numpy as np
import torch
import yaml
from ultralytics import YOLO


MLFLOW_TRACKING_URI = "https://dagshub.com/islam_tb/firewatch-ci.mlflow"


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

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Authenticate
    mlflow.tracking.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment("firewatch-ci")
    with mlflow.start_run(run_name="smoke-train" if smoke else "full-train") as run:
        try:
            mlflow.log_params(
                {
                    **cfg["model"],
                    **cfg["train"],
                    "epochs": epochs,
                    "smoke_run": smoke,
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
                project=str(out_dir_abs),
                name="train",
                exist_ok=True,
                seed=cfg["train"]["seed"],
                save_period=10,
                mosaic=cfg["train"]["mosaic"],
                hsv_h=cfg["train"]["hsv_h"],
                hsv_s=cfg["train"]["hsv_s"],
                hsv_v=cfg["train"]["hsv_v"],
                fliplr=cfg["train"]["fliplr"],
                degrees=cfg["train"]["degrees"],
                scale=cfg["train"]["scale"],
                translate=cfg["train"]["translate"],
                cls=cfg["train"]["cls"],
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

            csv_file = out_dir_abs / "train" / "results.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)

                for _, row in df.iterrows():
                    step = int(row["epoch"])

                    mlflow.log_metric(
                        "train_box_loss", float(row["train/box_loss"]), step=step
                    )

                    mlflow.log_metric(
                        "val_box_loss", float(row["val/box_loss"]), step=step
                    )

                    mlflow.log_metric(
                        "mAP50", float(row["metrics/mAP50(B)"]), step=step
                    )

                    mlflow.log_metric(
                        "precision", float(row["metrics/precision(B)"]), step=step
                    )
                    mlflow.log_metric(
                        "recall", float(row["metrics/recall(B)"]), step=step
                    )
                print("Successfully logged training metrics from CSV to MLflow.")
            else:
                print(f"Expected CSV file not found at: {csv_file}")

        except Exception as e:
            print(f"Training failed: {e}")
            mlflow.set_tag("status", "failed")
            raise


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
