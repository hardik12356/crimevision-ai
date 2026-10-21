
from pathlib import Path
from ultralytics import RTDETR


class ObjectDetector:
    """Detect objects in images using a pretrained RT-DETR model."""

    def __init__(self, model_path="rtdetr-l.pt", confidence_threshold=0.25):
        self.model = RTDETR(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, image_path):
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        results = self.model.predict(
            source=str(path),
            conf=self.confidence_threshold,
            verbose=False,
            device="cpu",
        )

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())

                detections.append({
                    "label": self.model.names[class_id],
                    "class_id": class_id,
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox_xyxy": [
                        round(float(value), 2)
                        for value in box.xyxy[0].tolist()
                    ],
                })

        return {
            "image": path.name,
            "model": "RT-DETR",
            "person_count": sum(
                item["label"] == "person" for item in detections
            ),
            "detection_count": len(detections),
            "detections": detections,
        }