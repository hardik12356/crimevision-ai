
"""
Vision Service
Coordinates image analysis for CrimeVision AI.
Activities 2.1 and 2.2.
"""

from pathlib import Path
from typing import Any
from app.services.scene_classifier import SceneClassifier
from app.services.object_detector import ObjectDetector
from app.services.image_captioner import ImageCaptioner
from app.services.groq_service import GroqService


class VisionService:
    """Orchestrate image-analysis components."""


    def __init__(
        self,
        object_detector: ObjectDetector | None = None,
        scene_classifier: SceneClassifier | None = None,
        image_captioner: ImageCaptioner | None = None,
        groq_service: GroqService | None = None,
    ):
        self.object_detector = object_detector or ObjectDetector()
        self.scene_classifier = scene_classifier
        self.image_captioner = image_captioner
        self.groq_service = groq_service or GroqService()

    def detect_objects(
        self,
        image_path: str,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Detect objects and return structured findings."""

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        # Apply the requested confidence threshold.
        original_threshold = self.object_detector.confidence_threshold
        try:
            self.object_detector.confidence_threshold = confidence_threshold
            detections = self.object_detector.detect(str(path))
        finally:
            self.object_detector.confidence_threshold = original_threshold

        # This is a count of person-labelled boxes, not verified
        # unique individuals. Overlapping detections may be duplicates.
        return {
            "image": path.name,
            "model": "RT-DETR",
            "detection_count": detections["detection_count"],
            "person_detection_count": detections["person_count"],
            "detections": detections["detections"],
            "count_note": (
                "Person count represents detected person-labelled boxes. "
                "It may overcount individuals when boxes overlap."
            ),
        }



    
    def analyze_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Run object detection, scene classification and image captioning."""

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        findings = self.detect_objects(
            str(path), confidence_threshold=confidence_threshold
        )

        if self.scene_classifier is None:
            self.scene_classifier = SceneClassifier()
        scene_result = self.scene_classifier.classify(str(path))

        if self.image_captioner is None:
            self.image_captioner = ImageCaptioner()
        caption_result = self.image_captioner.caption(str(path))

        groq_result = self.groq_service.analyze_image_findings({
            "object_detection": findings,
            "scene_classification": scene_result,
            "image_caption": caption_result,
        })

        return {
            "image": path.name,
            "object_detection": findings,
            "scene_classification": scene_result,
            "image_caption": caption_result,
            "scenario_hypotheses": {
                "status": "pending",
                "items": [],
                "note": "Hypotheses are possibilities, not established facts.",
            },
            "llm_analysis": groq_result,
        }