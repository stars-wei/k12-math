BEGIN;

CREATE SCHEMA IF NOT EXISTS observability;

CREATE TABLE IF NOT EXISTS observability.prompt_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_name TEXT NOT NULL,
    version TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    json_schema JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prompt_versions_name_not_blank
        CHECK (length(btrim(prompt_name)) > 0),
    CONSTRAINT prompt_versions_version_not_blank
        CHECK (length(btrim(version)) > 0),
    CONSTRAINT prompt_versions_name_version_unique
        UNIQUE (prompt_name, version)
);

CREATE TABLE IF NOT EXISTS observability.grading_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,

    input_kind TEXT NOT NULL,
    original_filename TEXT,
    content_type TEXT,
    file_size BIGINT,
    image_sha256 TEXT,

    model_name TEXT,
    application_version TEXT,
    status TEXT NOT NULL DEFAULT 'processing',
    overall_verdict TEXT,
    total_score NUMERIC(8, 2),
    max_total_score NUMERIC(8, 2),
    score_final BOOLEAN,
    error_message TEXT,

    CONSTRAINT grading_runs_trace_id_unique UNIQUE (trace_id),
    CONSTRAINT grading_runs_input_kind_valid
        CHECK (input_kind IN ('image', 'text')),
    CONSTRAINT grading_runs_status_valid
        CHECK (status IN ('processing', 'completed', 'failed')),
    CONSTRAINT grading_runs_file_size_nonnegative
        CHECK (file_size IS NULL OR file_size >= 0),
    CONSTRAINT grading_runs_image_sha256_valid
        CHECK (image_sha256 IS NULL OR image_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT grading_runs_score_nonnegative
        CHECK (total_score IS NULL OR total_score >= 0),
    CONSTRAINT grading_runs_max_score_nonnegative
        CHECK (max_total_score IS NULL OR max_total_score >= 0),
    CONSTRAINT grading_runs_score_not_above_max
        CHECK (
            total_score IS NULL
            OR max_total_score IS NULL
            OR total_score <= max_total_score
        ),
    CONSTRAINT grading_runs_completed_at_order
        CHECK (completed_at IS NULL OR completed_at >= created_at)
);

CREATE TABLE IF NOT EXISTS observability.grading_trace_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    grading_run_id UUID NOT NULL
        REFERENCES observability.grading_runs(id) ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL,
    stage TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::JSONB,
    duration_ms INTEGER,
    prompt_version_id UUID
        REFERENCES observability.prompt_versions(id) ON DELETE SET NULL,
    model_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT grading_trace_events_sequence_positive
        CHECK (sequence_number > 0),
    CONSTRAINT grading_trace_events_stage_not_blank
        CHECK (length(btrim(stage)) > 0),
    CONSTRAINT grading_trace_events_duration_nonnegative
        CHECK (duration_ms IS NULL OR duration_ms >= 0),
    CONSTRAINT grading_trace_events_run_sequence_unique
        UNIQUE (grading_run_id, sequence_number)
);

CREATE TABLE IF NOT EXISTS observability.grading_step_evaluations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    grading_run_id UUID NOT NULL
        REFERENCES observability.grading_runs(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    continuity_status TEXT,
    mathematical_validity TEXT,
    evaluation_status TEXT,
    step_score NUMERIC(8, 2),
    max_score NUMERIC(8, 2),
    diagnostic_message TEXT,
    feedback TEXT,
    error_category TEXT,
    model_output JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT grading_step_evaluations_step_positive
        CHECK (step_number > 0),
    CONSTRAINT grading_step_evaluations_continuity_valid
        CHECK (
            continuity_status IS NULL
            OR continuity_status IN (
                'complete',
                'acceptable_omission',
                'ambiguous',
                'logical_break'
            )
        ),
    CONSTRAINT grading_step_evaluations_math_validity_valid
        CHECK (
            mathematical_validity IS NULL
            OR mathematical_validity IN ('valid', 'invalid', 'unknown')
        ),
    CONSTRAINT grading_step_evaluations_status_valid
        CHECK (
            evaluation_status IS NULL
            OR evaluation_status IN (
                'passed',
                'passed_with_note',
                'unverified',
                'failed'
            )
        ),
    CONSTRAINT grading_step_evaluations_score_nonnegative
        CHECK (step_score IS NULL OR step_score >= 0),
    CONSTRAINT grading_step_evaluations_max_score_nonnegative
        CHECK (max_score IS NULL OR max_score >= 0),
    CONSTRAINT grading_step_evaluations_score_not_above_max
        CHECK (
            step_score IS NULL
            OR max_score IS NULL
            OR step_score <= max_score
        ),
    CONSTRAINT grading_step_evaluations_run_step_unique
        UNIQUE (grading_run_id, step_number)
);

CREATE INDEX IF NOT EXISTS grading_runs_created_at_idx
    ON observability.grading_runs (created_at DESC);

CREATE INDEX IF NOT EXISTS grading_runs_image_sha256_idx
    ON observability.grading_runs (image_sha256)
    WHERE image_sha256 IS NOT NULL;

CREATE INDEX IF NOT EXISTS grading_runs_status_idx
    ON observability.grading_runs (status);

CREATE INDEX IF NOT EXISTS grading_trace_events_run_stage_idx
    ON observability.grading_trace_events (grading_run_id, stage);

CREATE INDEX IF NOT EXISTS grading_trace_events_prompt_version_idx
    ON observability.grading_trace_events (prompt_version_id)
    WHERE prompt_version_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS grading_step_evaluations_run_status_idx
    ON observability.grading_step_evaluations (
        grading_run_id,
        evaluation_status
    );

COMMENT ON SCHEMA observability IS
    'OCR、模型调用与批改过程的可追踪运行记录。';

COMMENT ON TABLE observability.grading_runs IS
    '每次批改请求的总记录；只保存图片元数据和哈希，不保存图片内容。';

COMMENT ON TABLE observability.grading_trace_events IS
    '一次批改在 OCR、DeepSeek、标准化、评分和 API 响应等阶段的 JSON 快照。';

COMMENT ON TABLE observability.grading_step_evaluations IS
    '按步骤保存连续性、数学有效性、评分状态和诊断信息。';

COMMENT ON TABLE observability.prompt_versions IS
    '保存参与模型调用的提示词与 JSON Schema 版本。';

COMMIT;
