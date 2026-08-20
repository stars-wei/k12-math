BEGIN;

ALTER TABLE observability.grading_step_evaluations
    ADD COLUMN IF NOT EXISTS ocr_agreement TEXT,
    ADD COLUMN IF NOT EXISTS secondary_ocr_text TEXT,
    ADD COLUMN IF NOT EXISTS verification_message TEXT,
    ADD COLUMN IF NOT EXISTS ocr_fix_suggestion TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'grading_step_evaluations_ocr_agreement_valid'
          AND conrelid = 'observability.grading_step_evaluations'::regclass
    ) THEN
        ALTER TABLE observability.grading_step_evaluations
            ADD CONSTRAINT grading_step_evaluations_ocr_agreement_valid
            CHECK (
                ocr_agreement IS NULL
                OR ocr_agreement IN ('agree', 'disagree', 'uncertain', 'not_checked')
            );
    END IF;
END
$$;

COMMENT ON COLUMN observability.grading_step_evaluations.ocr_agreement IS
    'PaddleOCR 与 DeepSeek-OCR 对本步数学实质的一致性。';

COMMENT ON COLUMN observability.grading_step_evaluations.secondary_ocr_text IS
    '复核 OCR 中与本步对应的原始识别文本。';

COMMENT ON COLUMN observability.grading_step_evaluations.verification_message IS
    '双 OCR 一致、冲突或无法对齐的结构化说明。';

COMMENT ON COLUMN observability.grading_step_evaluations.ocr_fix_suggestion IS
    '模型建议的 OCR 修正文本；只作为诊断证据，不单独决定扣分。';

COMMIT;
