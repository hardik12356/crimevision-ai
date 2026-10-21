
from types import SimpleNamespace
from unittest.mock import patch

from app.services.groq_service import GroqService


def test_groq_service_uses_mocked_response():
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Summary: Unverified model outputs received."
                ),
                finish_reason="stop",
            )
        ]
    )

    with patch("app.services.groq_service.Groq") as mock_groq:
        mock_client = mock_groq.return_value
        mock_client.chat.completions.create.return_value = fake_response

        service = GroqService()
        result = service.analyze_image_findings({
            "object_detection": {
                "detections": [
                    {"label": "person", "confidence": 0.91}
                ]
            },
            "scene_classification": {
                "top_scene": "an indoor area"
            },
            "image_caption": {
                "caption": "A person near a table."
            },
        })

        assert result["model"] == "openai/gpt-oss-20b"
        assert "Unverified model outputs" in result["analysis"]
        assert result["truncated"] is False

        # Verify the mocked API client was called, not the real API.
        mock_client.chat.completions.create.assert_called_once()

    print("Mocked Groq service test passed. No real API request made.")


if __name__ == "__main__":
    test_groq_service_uses_mocked_response()