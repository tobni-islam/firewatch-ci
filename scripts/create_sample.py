import random
import shutil
from pathlib import Path

# Set random seed
random.seed(42)


def create_split(split_name, num_samples):
    # Source paths
    src_img_dir = Path(f"data/raw/{split_name}/images")
    src_lbl_dir = Path(f"data/raw/{split_name}/labels")

    # Destination paths
    dst_img_dir = Path(f"data/sample/{split_name}/images")
    dst_lbl_dir = Path(f"data/sample/{split_name}/labels")

    # Create destination directories
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    images = list(src_img_dir.glob("*.jpg"))
    if len(images) < num_samples:
        raise ValueError(
            f"Not enough images in {src_img_dir} to sample {num_samples} (found {len(images)})"
        )

    # Randomly sample the requested number of images
    sampled_images = random.sample(images, num_samples)

    print(f"Copying {num_samples} samples for '{split_name}' split...")
    for img_path in sampled_images:
        # Copy image
        shutil.copy(img_path, dst_img_dir / img_path.name)

        # Copy corresponding label text file if it exists
        lbl_name = img_path.stem + ".txt"
        src_lbl_path = src_lbl_dir / lbl_name
        if src_lbl_path.exists():
            shutil.copy(src_lbl_path, dst_lbl_dir / lbl_name)
        else:
            print(f"Warning: Label missing for {img_path.name}")


if __name__ == "__main__":
    create_split("train", 160)
    create_split("val", 40)
    print("Sample dataset creation complete!")
