
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class SceneClassifier:
    """Classify an image against a list of candidate scene descriptions."""

    MODEL_NAME = "openai/clip-vit-base-patch32"

    DEFAULT_SCENES = [
        "a photograph of a bedroom",
        "a photograph of a living room",
        "a photograph of a kitchen",
        "a photograph of an office",
        "a photograph of a road or street",
        "a photograph of a shop",
        "a photograph of a public park",
        "a photograph of a hallway",
        "a photograph of an outdoor area",
        "a photograph of an indoor area",
    ]

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def classify(
        self,
        image_path: str,
        candidate_scenes: list[str] | None = None,
    ) -> dict:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        scenes = candidate_scenes or self.DEFAULT_SCENES
        if not scenes:
            raise ValueError("Provide at least one candidate scene.")

        with Image.open(path) as source:
            image = source.convert("RGB")

        inputs = self.processor(
            text=scenes,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            scores = outputs.logits_per_image[0].softmax(dim=0)

        ranked = sorted(
            [
                {
                    "scene": scene,
                    "similarity_score": round(float(score), 4),
                }
                for scene, score in zip(scenes, scores.tolist())
            ],
            key=lambda item: item["similarity_score"],
            reverse=True,
        )

        return {
            "model": self.MODEL_NAME,
            "device": self.device,
            "top_scene": ranked[0]["scene"],
            "candidates": ranked,
            "score_note": (
                "Scores are relative across the supplied candidates; "
                "they are not calibrated probabilities."
            ),
        }