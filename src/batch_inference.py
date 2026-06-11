import argparse
import csv
import time
from pathlib import Path
import yaml
from ultralytics import YOLO


def run_batch_inference(weights: str, cfg: dict) -> None:
    test_imgs = Path(cfg["data"]["processed_dir"]) / "test" / "images"
    log_path = Path("logs/inference_log.csv")
    log_path.parent.mkdir(exist_ok=True)

    model = YOLO(weights)
    images = sorted(test_imgs.glob("*.jpg"))

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "predicted_class",
                "confidence",
                "num_detections",
                "timestamp",
            ],
        )
        writer.writeheader()
        for img_path in images:
            results = model.predict(
                str(img_path),
                conf=cfg["evaluate"]["conf_threshold"],
                verbose=False,
            )
            for result in results:
                if len(result.boxes) == 0:
                    writer.writerow(
                        {
                            "image": img_path.name,
                            "predicted_class": "none",
                            "confidence": 0.0,
                            "num_detections": 0,
                            "timestamp": round(time.time(), 2),
                        }
                    )
                else:
                    best_idx = result.boxes.conf.argmax()
                    best = result.boxes[best_idx]
                    cls_id = int(best.cls[0])
                    writer.writerow(
                        {
                            "image": img_path.name,
                            "predicted_class": cfg["model"]["class_names"][cls_id],
                            "confidence": round(float(best.conf[0]), 4),
                            "num_detections": len(result.boxes),
                            "timestamp": round(time.time(), 2),
                        }
                    )

    print(f"Inference log → {log_path}  ({len(images)} images)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--config", default="params.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_batch_inference(args.weights, cfg)
