import argparse
import json
from pathlib import Path

import cv2
import yaml


def check_image_label_pairs(raw_dir: Path) -> dict:
    images = sorted((raw_dir / "train" / "images").glob("*.jpg"))
    labels_dir = raw_dir / "train" / "labels"
    missing = [i.name for i in images if not (labels_dir / (i.stem + ".txt")).exists()]
    return {
        "check": "image_label_pairs",
        "total_images": len(images),
        "missing_labels": len(missing),
        "passed": len(missing) == 0,
        "sample_missing": missing[:5],
    }


def check_image_readable(raw_dir: Path, sample_size: int = 200) -> dict:
    images = sorted((raw_dir / "train" / "images").glob("*.jpg"))[:sample_size]
    corrupt = [i.name for i in images if cv2.imread(str(i)) is None]
    return {
        "check": "image_readable",
        "sample_checked": len(images),
        "corrupt": len(corrupt),
        "passed": len(corrupt) == 0,
        "sample_corrupt": corrupt[:5],
    }


def check_label_schema(raw_dir: Path) -> dict:
    labels = sorted((raw_dir / "train" / "labels").glob("*.txt"))
    bad = []
    for lbl in labels:
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if (
                    len(parts) != 5
                    or int(parts[0]) not in (0, 1)
                    or not all(0.0 <= float(v) <= 1.0 for v in parts[1:])
                ):
                    bad.append(lbl.name)
                    break
    return {
        "check": "label_schema",
        "total_labels": len(labels),
        "bad_files": len(bad),
        "passed": len(bad) == 0,
        "sample_bad": bad[:5],
    }


def check_class_distribution(raw_dir: Path) -> dict:
    labels = sorted((raw_dir / "train" / "labels").glob("*.txt"))
    counts = {0: 0, 1: 0}
    for lbl in labels:
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) in counts:
                    counts[int(parts[0])] += 1
    return {
        "check": "class_distribution",
        "fire_count": counts[0],
        "smoke_count": counts[1],
        "passed": counts[0] > 0 and counts[1] > 0,
    }


def check_row_count(raw_dir: Path, min_count: int = 1000) -> dict:
    n = len(list((raw_dir / "train" / "images").glob("*.jpg")))
    return {
        "check": "row_count",
        "count": n,
        "min_expected": min_count,
        "passed": n >= min_count,
    }


def run_validation(cfg: dict) -> dict:
    raw_dir = Path(cfg["data"]["raw_dir"])
    checks = [
        check_image_label_pairs(raw_dir),
        check_image_readable(raw_dir),
        check_label_schema(raw_dir),
        check_class_distribution(raw_dir),
        check_row_count(raw_dir),
    ]
    report = {"all_passed": all(c["passed"] for c in checks), "checks": checks}
    Path("data").mkdir(exist_ok=True)
    with open("data/validate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="params.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    report = run_validation(cfg)
    if not report["all_passed"]:
        raise SystemExit(1)
