import cv2
import os
import numpy as np
import csv
import bentoml
from bentoml.io import Image, JSON
from PIL import Image as PILImage
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel, Field


CLASS_NAMES = ["fire", "smoke"]
ALERT_THRESHOLD = 0.7
MODEL_PATH = "models/weights/train/weights/best.pt"


WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # http://localhost:3000/webhook_fire
WEBHOOK_LOG = Path("logs/webhook_alerts.csv")


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

    await _maybe_trigger_webhook(alert_level, detections)

    return {
        "detections": detections,
        "num_detections": len(detections),
        "alert_level": alert_level,
        "classes_detected": list({d["class"] for d in detections}),
    }


class FireAlert(BaseModel):
    alert_level: str = Field(..., pattern="^(HIGH|LOW)$")
    source: str = "unknown"
    detections: list = []
    timestamp: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )


def _log_alert(alert: "FireAlert") -> None:
    WEBHOOK_LOG.parent.mkdir(exist_ok=True)
    is_new = not WEBHOOK_LOG.exists()
    with open(WEBHOOK_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "alert_level", "source", "num_detections"])
        writer.writerow(
            [alert.timestamp, alert.alert_level, alert.source, len(alert.detections)]
        )


@svc.api(input=JSON(pydantic_model=FireAlert), output=JSON())
def webhook_fire(alert: FireAlert) -> dict:
    """Manually-testable alert endpoint. Logs to logs/webhook_alerts.csv."""
    _log_alert(alert)
    return {
        "status": "received",
        "alert_level": alert.alert_level,
        "logged_to": str(WEBHOOK_LOG),
    }


async def _maybe_trigger_webhook(alert_level: str, detections: list) -> None:
    """Fire-and-forget internal trigger. Never raises, never blocks /detect."""
    if not WEBHOOK_URL or alert_level != "HIGH":
        return
    payload = {
        "alert_level": alert_level,
        "source": "detect-endpoint",
        "detections": detections,
    }
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(WEBHOOK_URL, json=payload)
    except Exception as e:
        print(f"webhook call failed (non-fatal): {e}")
