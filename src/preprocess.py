import argparse
import json
import random
import shutil
from pathlib import Path

import yaml


def validate_annotation(lbl: Path) -> bool:
    """YOLO format: 5 values per line, class in {0, 1}, coords in [0, 1]."""
    try:
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                if int(parts[0]) not in (0, 1):
                    return False
                if not all(0.0 <= float(v) <= 1.0 for v in parts[1:]):
                    return False
        return True
    except (ValueError, IOError):
        return False


def preprocess(cfg: dict) -> None:
    raw = Path(cfg["data"]["raw_dir"])
    out = Path(cfg["data"]["processed_dir"])
    random.seed(cfg["train"]["seed"])

    images = sorted((raw / "train" / "images").glob("*.jpg"))
    valid, skipped = [], 0

    for img in images:
        lbl = raw / "train" / "labels" / (img.stem + ".txt")
        if lbl.exists() and validate_annotation(lbl):
            valid.append((img, lbl))
        else:
            skipped += 1

    images = sorted((raw / "val" / "images").glob("*.jpg"))

    for img in images:
        lbl = raw / "val" / "labels" / (img.stem + ".txt")
        if lbl.exists() and validate_annotation(lbl):
            valid.append((img, lbl))
        else:
            skipped += 1

    print(f"Valid: {len(valid)} | Skipped: {skipped}")
    random.shuffle(valid)

    n = len(valid)
    n_tr = int(n * cfg["data"]["train_ratio"])
    n_va = int(n * cfg["data"]["val_ratio"])
    splits = {
        "train": valid[:n_tr],
        "val": valid[n_tr : n_tr + n_va],
        "test": valid[n_tr + n_va :],
    }

    counts = {0: 0, 1: 0}
    for split, pairs in splits.items():
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            shutil.copy(img, out / split / "images" / img.name)
            shutil.copy(lbl, out / split / "labels" / lbl.name)
            with open(lbl) as f:
                for line in f:
                    counts[int(line.strip().split()[0])] += 1

    stats = {
        "total_valid": len(valid),
        "skipped": skipped,
        "splits": {k: len(v) for k, v in splits.items()},
        "class_counts": {"fire": counts[0], "smoke": counts[1]},
    }
    with open("data/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="params.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    preprocess(cfg)
