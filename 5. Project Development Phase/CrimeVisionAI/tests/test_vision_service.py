
import json

from app.services.vision_service import VisionService


def main():
    service = VisionService()
    result = service.analyze_image("evidence/test_image.jpg")

    print(json.dumps(result, indent=2))

    with open(
        "outputs/vision_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2)

    print("\nSaved to outputs/vision_analysis.json")


if __name__ == "__main__":
    main()