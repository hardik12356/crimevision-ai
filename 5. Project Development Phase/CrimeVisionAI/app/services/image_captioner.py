
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptioner:
    """Generate a natural-language caption for an image."""

    MODEL_NAME = "Salesforce/blip-image-captioning-base"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(self.MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.MODEL_NAME,
            use_safetensors=False,
        )
        self.model.to(self.device)
        self.model.eval()

    def caption(self, image_path: str) -> dict:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as source:
            image = source.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=60,
            )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True,
        )

        return {
            "model": self.MODEL_NAME,
            "caption": caption,
            "note": (
                "Automatically generated description; verify against "
                "the original evidence."
            ),
        }