import cv2
import numpy as np


def make_image(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_image_loads_correctly(tmp_path):
    p = tmp_path / "test.jpg"
    cv2.imwrite(str(p), make_image())
    img = cv2.imread(str(p))
    assert img is not None
    assert img.ndim == 3 and img.shape[2] == 3


def test_yolo_label_format_valid(tmp_path):
    lbl = tmp_path / "test.txt"
    lbl.write_text("0 0.512 0.341 0.124 0.089\n1 0.3 0.3 0.1 0.1\n")
    for line in lbl.read_text().strip().splitlines():
        parts = line.split()
        assert len(parts) == 5
        assert int(parts[0]) in (0, 1)
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_augmented_dir_structure(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    assert (tmp_path / "images").is_dir()
    assert (tmp_path / "labels").is_dir()
