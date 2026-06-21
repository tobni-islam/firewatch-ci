import cv2
import os
import numpy as np
import bentoml
from bentoml.io import Image, JSON
from PIL import Image as PILImage


CLASS_NAMES = ["fire", "smoke"]
ALERT_THRESHOLD = 0.7
MODEL_PATH = "models/weights/train/weights/best.pt"


class FireWatchRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        from ultralytics import YOLO

        bento_model = bentoml.models.get("firewatch-detector:latest")
        model_path = os.path.join(bento_model.path, "best.pt")
        self.model = YOLO(model_path)

    @bentoml.Runnable.method(batchable=False)
    def detect(self, image_bgr: np.ndarray):
        return self.model.predict(image_bgr, verbose=False)


runner = bentoml.Runner(FireWatchRunnable, name="firewatch-runner")
svc = bentoml.Service("firewatch-detector", runners=[runner])


@svc.api(input=Image(), output=JSON())
async def detect(image: PILImage.Image) -> dict:
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = await runner.detect.async_run(img_bgr)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append(
                {
                    "class": CLASS_NAMES[cls_id],
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 2) for v in box.xyxy[0].tolist()],
                }
            )

    alert_level = (
        "HIGH" if any(d["confidence"] >= ALERT_THRESHOLD for d in detections) else "LOW"
    )

    return {
        "detections": detections,
        "num_detections": len(detections),
        "alert_level": alert_level,
        "classes_detected": list({d["class"] for d in detections}),
    }
