# Development Log

## 2026-08-13 — Codex

### Repository baseline and privacy audit

- Audited the existing `math-v0.1` codebase before creating its public Git baseline.
- Confirmed that API keys and Neo4j credentials are read from environment variables rather than hard-coded in source files.
- Added `.gitignore` rules for environment files, Python cache files, generated HTML output, and editor-specific files.
- Confirmed the current implementation already supports OCR input, compound-task recognition, and graph-backed execution for quadratic-function axis and extremum tasks.
- Identified that the next version should improve result semantics, remove duplicate answers, consistently render inline mathematics, make task relationships explicit, surface unsupported tasks clearly, reuse shared intermediate results, and improve user-facing failure messages.

### Version 1 publication

- Created the public GitHub repository `stars-wei/k12-math`.
- Prepared the audited `math-v0.1` codebase as the first published version, with credentials and generated outputs excluded from version control.

## 2026-08-13 — Codex

### Version 2 execution and presentation improvements

- Created the `v0.2` development branch from the public `v0.1.0` baseline.
- Added an auditable Neo4j migration for the independent quadratic-function vertex Task and its graph-backed completing-square strategy.
- Added verified reusable facts (`axis`, `vertex_value`, and extremum facts), so later requested tasks can reuse earlier verified results instead of recomputing them.
- Replaced terse or duplicated final answers with one semantic answer per task, such as “顶点为 …” and “对称轴为 …”.
- Added the tracked `templates/result.html` source template and enabled MathJax rendering in operation descriptions as well as formulas.
- Made unsupported tasks explicitly state that no answer was produced for that requirement.
- Added safe, service-specific user messages for OCR, DeepSeek, Neo4j, and strategy-execution failures; browser disconnects no longer produce a server traceback.
- Expanded regression coverage for vertex recognition, fact reuse, semantic answers, and unsupported-task messaging.

### Source-tree refactor

- Moved executable Python modules into `src/`, tests into `tests/`, the development log into `docs/`, and Neo4j migrations into `graph/migrations/`.
- Kept the source layout intentionally flat: the project is not yet distributed as an installable Python package, so there is no additional package-name directory under `src/`.
