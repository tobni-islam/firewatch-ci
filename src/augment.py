import argparse
import shutil
from pathlib import Path
import albumentations as A
import cv2
import yaml


def load_labels(path):
    boxes = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 5:
                boxes.append([int(p[0])] + [float(v) for v in p[1:]])
    return boxes


def save_labels(path, boxes):
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def build_transform():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.4, p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=40, p=0.5
            ),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
            A.RandomRain(p=0.15),
            A.Blur(blur_limit=5, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


def augment(cfg, n=2):
    src_imgs = Path(cfg["data"]["processed_dir"]) / "train" / "images"
    src_lbls = Path(cfg["data"]["processed_dir"]) / "train" / "labels"
    out_imgs = Path("data/augmented/train/images")
    out_lbls = Path("data/augmented/train/labels")
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)
    transform = build_transform()
    images = sorted(src_imgs.glob("*.jpg"))
    aug_count = 0
    for img_path in images:
        lbl_path = src_lbls / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        shutil.copy(img_path, out_imgs / img_path.name)
        shutil.copy(lbl_path, out_lbls / lbl_path.name)
        raw = load_labels(lbl_path)
        if not raw:
            continue
        img_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        class_labels = [b[0] for b in raw]
        bboxes = [[b[1], b[2], b[3], b[4]] for b in raw]
        for i in range(n):
            try:
                res = transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
                if not res["bboxes"]:
                    continue
                aug_bgr = cv2.cvtColor(res["image"], cv2.COLOR_RGB2BGR)
                stem = f"{img_path.stem}_aug{i}"
                cv2.imwrite(str(out_imgs / f"{stem}.jpg"), aug_bgr)
                new_boxes = [
                    [cls, *bb] for cls, bb in zip(res["class_labels"], res["bboxes"])
                ]
                save_labels(out_lbls / f"{stem}.txt", new_boxes)
                aug_count += 1
            except Exception as e:
                print(f"  skip {img_path.stem} aug{i}: {e}")
    print(f"originals: {len(images)}  new: {aug_count}  total: {len(images)+aug_count}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="params.yaml")
    p.add_argument("--n", type=int)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    n = args.n or cfg["augment"]["n_per_image"]
    augment(cfg, n=n)
