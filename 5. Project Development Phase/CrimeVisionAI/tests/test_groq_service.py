
import json

from app.services.groq_service import GroqService


def test_groq_analysis():
    service = GroqService()

    sample_findings = {
        "object_detection": {
            "detection_count": 2,
            "person_detection_count": 1,
            "detections": [
                {"label": "person", "confidence": 0.91},
                {"label": "book", "confidence": 0.87},
            ],
            "count_note": "Counts may be affected by overlapping boxes.",
        },
        "scene_classification": {
            "top_scene": "a photograph of an indoor area",
            "score_note": "Scores are not calibrated probabilities.",
        },
        "image_caption": {
            "caption": "A person standing near a table with a book.",
            "note": "Automatically generated; verify against the image.",
        },
    }

    result = service.analyze_image_findings(sample_findings)

    assert result["model"] == GroqService.MODEL_NAME
    assert isinstance(result["analysis"], str)
    assert len(result["analysis"].strip()) > 50
    assert "note" in result

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nGroq service test passed.")


if __name__ == "__main__":
    test_groq_analysis()