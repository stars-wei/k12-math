"""Tests for best-effort PostgreSQL grading traces."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from grading_trace import PostgresTraceRepository, TraceRecorder, image_metadata


class FakeRepository:
    def __init__(self) -> None:
        self.run_id = uuid4()
        self.runs: list[dict] = []
        self.events: list[dict] = []
        self.prompts: list[dict] = []
        self.steps: list[dict] = []
        self.completed: list[dict] = []
        self.failed: list[str] = []

    def create_run(self, values):
        self.runs.append(values)
        return self.run_id

    def ensure_prompt_version(self, prompt_name, version, system_prompt, json_schema):
        self.prompts.append(
            {
                "prompt_name": prompt_name,
                "version": version,
                "system_prompt": system_prompt,
                "json_schema": json_schema,
            }
        )
        return uuid4()

    def add_event(
        self,
        run_id,
        sequence_number,
        stage,
        payload,
        duration_ms=None,
        prompt_version_id=None,
        model_name=None,
    ):
        self.events.append(
            {
                "run_id": run_id,
                "sequence_number": sequence_number,
                "stage": stage,
                "payload": payload,
                "duration_ms": duration_ms,
                "prompt_version_id": prompt_version_id,
                "model_name": model_name,
            }
        )

    def save_step(self, run_id, step):
        self.steps.append({"run_id": run_id, "step": step})

    def complete_run(self, run_id, result):
        self.completed.append({"run_id": run_id, "result": result})

    def fail_run(self, run_id, message):
        self.failed.append(message)


class TraceRecorderTests(unittest.TestCase):
    def test_image_metadata_contains_digest_but_not_image_bytes(self) -> None:
        metadata = image_metadata("question.png", "image/png", b"same-image")

        self.assertEqual(metadata["original_filename"], "question.png")
        self.assertEqual(metadata["content_type"], "image/png")
        self.assertEqual(metadata["file_size"], 10)
        self.assertEqual(len(metadata["image_sha256"]), 64)
        self.assertNotIn("bytes", metadata)
        self.assertNotIn("content", metadata)

    def test_complete_persists_events_steps_and_summary(self) -> None:
        repository = FakeRepository()
        recorder = TraceRecorder(repository)
        report = {
            "overall_verdict": "CORRECT",
            "total_score": 2,
            "max_total_score": 2,
            "score_final": True,
            "steps_evaluation": [
                {
                    "step_index": 1,
                    "raw_text": "a=1",
                    "is_valid": True,
                    "step_score": 2,
                    "max_score": 2,
                    "evaluation_status": "passed",
                    "continuity_status": "acceptable_omission",
                }
            ],
        }

        self.assertTrue(recorder.start("text"))
        recorder.event("request_input", {"question": "sample"})
        recorder.complete({"grading_report": report})

        self.assertEqual([event["sequence_number"] for event in repository.events], [1, 2])
        self.assertEqual(repository.events[-1]["stage"], "api_response")
        self.assertEqual(repository.steps[0]["step"]["raw_text"], "a=1")
        self.assertEqual(repository.completed[0]["result"], report)
        self.assertTrue(recorder.finalized)

    def test_model_request_creates_reusable_prompt_version(self) -> None:
        repository = FakeRepository()
        recorder = TraceRecorder(repository)
        recorder.start("text")
        payload = {
            "tool_name": "normalize_steps",
            "model": "deepseek-test",
            "system": "system prompt",
            "parameters": {"type": "object"},
            "user": "student work",
        }

        recorder.model_event("deepseek_request", payload)
        recorder.model_event("deepseek_request", payload)

        self.assertEqual(len(repository.prompts), 1)
        self.assertIsNotNone(repository.events[0]["prompt_version_id"])
        self.assertEqual(repository.events[0]["model_name"], "deepseek-test")

    def test_missing_configuration_disables_repository(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(PostgresTraceRepository.from_env())


if __name__ == "__main__":
    unittest.main()
