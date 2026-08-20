"""Best-effort PostgreSQL persistence for grading diagnostics.

The tracing layer must never make a grading request fail.  It stores text and
JSON snapshots, but only stores image metadata and a SHA-256 digest—not image
bytes or a filesystem copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    psycopg = None
    Jsonb = None


def _json_value(value: Any) -> Any:
    """Return a JSON-compatible copy of an arbitrary diagnostic value."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def image_metadata(filename: str, content_type: str, image: bytes) -> dict[str, Any]:
    """Build the only image data that may be persisted by this application."""
    return {
        "original_filename": filename or None,
        "content_type": content_type or None,
        "file_size": len(image),
        "image_sha256": hashlib.sha256(image).hexdigest(),
    }


class PostgresTraceRepository:
    """Small psycopg repository for the observability schema."""

    def __init__(self, dsn: str | None = None, **connection_kwargs: Any) -> None:
        if psycopg is None:
            raise RuntimeError("缺少 psycopg，无法写入 PostgreSQL 运行记录。")
        self.dsn = dsn
        self.connection_kwargs = connection_kwargs

    @classmethod
    def from_env(cls) -> PostgresTraceRepository | None:
        """Create a repository when tracing is configured, otherwise disable it."""
        enabled = os.getenv("POSTGRES_TRACE_ENABLED", "true").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None

        dsn = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")
        if dsn:
            return cls(dsn=dsn)

        password = os.getenv("POSTGRES_PASSWORD")
        if not password:
            return None
        return cls(
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "demo"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=password,
        )

    def _connect(self):
        if self.dsn:
            return psycopg.connect(self.dsn, connect_timeout=3)
        return psycopg.connect(connect_timeout=3, **self.connection_kwargs)

    def create_run(self, values: dict[str, Any]) -> tuple[UUID, UUID]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO observability.grading_runs (
                    input_kind, original_filename, content_type, file_size,
                    image_sha256, model_name, application_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, trace_id
                """,
                (
                    values["input_kind"],
                    values.get("original_filename"),
                    values.get("content_type"),
                    values.get("file_size"),
                    values.get("image_sha256"),
                    values.get("model_name"),
                    values.get("application_version"),
                ),
            )
            return cursor.fetchone()

    def ensure_prompt_version(
        self,
        prompt_name: str,
        version: str,
        system_prompt: str,
        json_schema: dict[str, Any],
    ) -> UUID:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO observability.prompt_versions (
                    prompt_name, version, system_prompt, json_schema
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (prompt_name, version) DO UPDATE
                SET system_prompt = EXCLUDED.system_prompt,
                    json_schema = EXCLUDED.json_schema
                RETURNING id
                """,
                (prompt_name, version, system_prompt, Jsonb(_json_value(json_schema))),
            )
            return cursor.fetchone()[0]

    def add_event(
        self,
        run_id: UUID,
        sequence_number: int,
        stage: str,
        payload: Any,
        duration_ms: int | None = None,
        prompt_version_id: UUID | None = None,
        model_name: str | None = None,
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO observability.grading_trace_events (
                    grading_run_id, sequence_number, stage, payload,
                    duration_ms, prompt_version_id, model_name
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    sequence_number,
                    stage,
                    Jsonb(_json_value(payload)),
                    duration_ms,
                    prompt_version_id,
                    model_name,
                ),
            )

    def save_step(self, run_id: UUID, step: dict[str, Any]) -> None:
        is_valid = step.get("is_valid")
        mathematical_validity = (
            "unknown" if is_valid is None else "valid" if is_valid else "invalid"
        )
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO observability.grading_step_evaluations (
                    grading_run_id, step_number, raw_text, continuity_status,
                    mathematical_validity, evaluation_status, step_score,
                    max_score, diagnostic_message, feedback, error_category,
                    ocr_agreement, secondary_ocr_text, verification_message,
                    ocr_fix_suggestion, model_output
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (grading_run_id, step_number) DO UPDATE SET
                    raw_text = EXCLUDED.raw_text,
                    continuity_status = EXCLUDED.continuity_status,
                    mathematical_validity = EXCLUDED.mathematical_validity,
                    evaluation_status = EXCLUDED.evaluation_status,
                    step_score = EXCLUDED.step_score,
                    max_score = EXCLUDED.max_score,
                    diagnostic_message = EXCLUDED.diagnostic_message,
                    feedback = EXCLUDED.feedback,
                    error_category = EXCLUDED.error_category,
                    ocr_agreement = EXCLUDED.ocr_agreement,
                    secondary_ocr_text = EXCLUDED.secondary_ocr_text,
                    verification_message = EXCLUDED.verification_message,
                    ocr_fix_suggestion = EXCLUDED.ocr_fix_suggestion,
                    model_output = EXCLUDED.model_output
                """,
                (
                    run_id,
                    int(step.get("step_index") or step.get("step_number") or 1),
                    str(step.get("raw_text") or step.get("step_text") or ""),
                    step.get("continuity_status"),
                    mathematical_validity,
                    step.get("evaluation_status"),
                    step.get("step_score"),
                    step.get("max_score"),
                    step.get("error_detail") or step.get("diagnostic_message"),
                    step.get("feedback"),
                    step.get("error_category"),
                    step.get("ocr_agreement"),
                    step.get("secondary_ocr_evidence"),
                    step.get("verification_message"),
                    step.get("ocr_fix_suggestion"),
                    Jsonb(_json_value(step)),
                ),
            )

    def complete_run(self, run_id: UUID, result: dict[str, Any]) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE observability.grading_runs
                SET status = 'completed', completed_at = now(),
                    overall_verdict = %s, total_score = %s,
                    max_total_score = %s, score_final = %s,
                    error_message = NULL
                WHERE id = %s
                """,
                (
                    result.get("overall_verdict"),
                    result.get("total_score"),
                    result.get("max_total_score"),
                    result.get("score_final"),
                    run_id,
                ),
            )

    def fail_run(self, run_id: UUID, message: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE observability.grading_runs
                SET status = 'failed', completed_at = now(), error_message = %s
                WHERE id = %s
                """,
                (message, run_id),
            )


@dataclass
class TraceRecorder:
    """Request-scoped facade that converts persistence failures into warnings."""

    repository: PostgresTraceRepository
    run_id: UUID | None = None
    trace_id: UUID | None = None
    sequence_number: int = 0
    finalized: bool = False
    prompt_versions: dict[str, UUID] = field(default_factory=dict)

    def _warning(self, action: str, error: Exception) -> None:
        print(f"⚠️ [TRACE] PostgreSQL {action}失败，不影响本次处理：{error}", file=sys.stderr, flush=True)

    def start(self, input_kind: str, **metadata: Any) -> bool:
        values = {
            "input_kind": input_kind,
            "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "application_version": os.getenv("APP_VERSION"),
            **metadata,
        }
        try:
            created = self.repository.create_run(values)
            if isinstance(created, tuple):
                self.run_id, self.trace_id = created
            else:  # Keep small test doubles and custom repositories compatible.
                self.run_id = created
                self.trace_id = created
            return True
        except Exception as error:
            self._warning("创建运行记录", error)
            return False

    def event(
        self,
        stage: str,
        payload: Any,
        duration_ms: int | None = None,
        prompt_version_id: UUID | None = None,
        model_name: str | None = None,
    ) -> None:
        if self.run_id is None or self.finalized:
            return
        next_sequence = self.sequence_number + 1
        try:
            self.repository.add_event(
                self.run_id,
                next_sequence,
                stage,
                payload,
                duration_ms,
                prompt_version_id,
                model_name,
            )
            self.sequence_number = next_sequence
        except Exception as error:
            self._warning(f"写入 {stage} 事件", error)

    def model_event(self, stage: str, payload: dict[str, Any], duration_ms: int | None = None) -> None:
        """Accept events emitted by DeepSeekClient and version its prompt."""
        prompt_version_id = None
        if stage == "deepseek_request" and self.run_id is not None:
            prompt_name = str(payload.get("tool_name") or "unknown_tool")
            system_prompt = str(payload.get("system") or "")
            schema = payload.get("parameters") or {}
            canonical = json.dumps(schema, sort_keys=True, ensure_ascii=False)
            version = hashlib.sha256(
                f"{prompt_name}\n{system_prompt}\n{canonical}".encode("utf-8")
            ).hexdigest()[:16]
            cache_key = f"{prompt_name}:{version}"
            prompt_version_id = self.prompt_versions.get(cache_key)
            if prompt_version_id is None:
                try:
                    prompt_version_id = self.repository.ensure_prompt_version(
                        prompt_name, version, system_prompt, schema
                    )
                    self.prompt_versions[cache_key] = prompt_version_id
                except Exception as error:
                    self._warning("保存提示词版本", error)
        self.event(
            stage,
            payload,
            duration_ms=duration_ms,
            prompt_version_id=prompt_version_id,
            model_name=str(payload.get("model") or "") or None,
        )

    @staticmethod
    def _grading_result(payload: dict[str, Any]) -> dict[str, Any]:
        report = payload.get("grading_report")
        return report if isinstance(report, dict) else payload

    def complete(self, payload: dict[str, Any]) -> None:
        if self.run_id is None or self.finalized:
            return
        result = self._grading_result(payload)
        self.event("api_response", payload)
        steps = result.get("steps_evaluation", [])
        if not isinstance(steps, list):
            steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            try:
                self.repository.save_step(self.run_id, step)
            except Exception as error:
                self._warning("保存逐步批改结果", error)
        try:
            self.repository.complete_run(self.run_id, result)
        except Exception as error:
            self._warning("完成运行记录", error)
        self.finalized = True

    def fail(self, error: Exception | str) -> None:
        if self.run_id is None or self.finalized:
            return
        message = str(error)
        self.event("error", {"message": message})
        try:
            self.repository.fail_run(self.run_id, message)
        except Exception as persistence_error:
            self._warning("标记失败运行", persistence_error)
        self.finalized = True


def create_trace_recorder() -> TraceRecorder | None:
    """Return a configured recorder; missing configuration simply disables tracing."""
    try:
        repository = PostgresTraceRepository.from_env()
        return TraceRecorder(repository) if repository is not None else None
    except Exception as error:
        print(f"⚠️ [TRACE] PostgreSQL 追踪未启用：{error}", file=sys.stderr, flush=True)
        return None
