import argparse
import json
from pathlib import Path
import yaml
from ultralytics import YOLO


def train(cfg: dict, epochs: int, smoke: bool = False) -> None:
    dataset = "data/sample/dataset.yaml" if smoke else "data/dataset.yaml"
    out_dir = (
        str(Path("models/smoke_weights").resolve())
        if smoke
        else str(Path("models/weights").resolve())
    )

    metrics_file = (
        "metrics/smoke_results.json" if smoke else "metrics/train_results.json"
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    model = YOLO(f"{cfg['model']['architecture']}.pt")

    results = model.train(
        data=dataset,
        epochs=epochs,
        imgsz=cfg["model"]["img_size"],
        batch=cfg["smoke_train"]["batch_size"] if smoke else cfg["train"]["batch_size"],
        project=out_dir,
        name="train",
        exist_ok=True,
        seed=cfg["train"]["seed"],
    )

    metrics = {
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Done — mAP50: {metrics['mAP50']:.3f}")


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
