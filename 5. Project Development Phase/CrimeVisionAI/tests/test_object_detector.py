
import json
from pathlib import Path

from app.services.object_detector import ObjectDetector


def main():
    image_path = Path("evidence/test_image.jpg")

    detector = ObjectDetector()
    result = detector.detect(str(image_path))

    print(json.dumps(result, indent=2))

    output_path = Path("outputs/detection_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()