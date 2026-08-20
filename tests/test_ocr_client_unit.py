"""Unit tests for dual-model OCR orchestration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ocr_client import DEEPSEEK_OCR_MODEL, PADDLE_OCR_MODEL, OcrClient


class DualOcrTests(unittest.TestCase):
    def test_pair_runs_paddle_and_deepseek(self) -> None:
        client = OcrClient(api_key="test-key")

        def transcribe(_image, _media_type, model):
            return f"text from {model}"

        with patch.object(client, "transcribe", side_effect=transcribe) as mocked:
            result = client.transcribe_pair(b"image", "image/jpeg")

        self.assertEqual(result["primary_model"], PADDLE_OCR_MODEL)
        self.assertEqual(result["secondary_model"], DEEPSEEK_OCR_MODEL)
        self.assertEqual(result["primary_text"], f"text from {PADDLE_OCR_MODEL}")
        self.assertEqual(result["secondary_text"], f"text from {DEEPSEEK_OCR_MODEL}")
        self.assertIsNone(result["secondary_error"])
        self.assertEqual(mocked.call_count, 2)

    def test_secondary_failure_keeps_primary_result(self) -> None:
        client = OcrClient(api_key="test-key")

        def transcribe(_image, _media_type, model):
            if model == DEEPSEEK_OCR_MODEL:
                raise RuntimeError("secondary unavailable")
            return "primary text"

        with patch.object(client, "transcribe", side_effect=transcribe):
            result = client.transcribe_pair(b"image", "image/jpeg")

        self.assertEqual(result["primary_text"], "primary text")
        self.assertIsNone(result["secondary_text"])
        self.assertIn("secondary unavailable", result["secondary_error"])


if __name__ == "__main__":
    unittest.main()
