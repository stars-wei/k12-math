"""Constrained Task and Strategy selection using DeepSeek plus Neo4j."""

from __future__ import annotations

import json

from deepseek_client import DeepSeekClient
from problem import Candidate, Classification, Problem
from solve import OPERATIONS, load_operations, query_neo4j


TASK_KEYWORDS = {
    "quadratic-function-axis": ("对称轴",),
    "quadratic-function-extremum": ("最值", "最大值", "最小值", "极值", "极大值", "极小值"),
}


def load_task_candidates(url: str, password: str) -> list[Candidate]:
    rows = query_neo4j(
        url,
        password,
        """
        MATCH (task:Task)
        RETURN task.id AS id, task.name AS name,
               coalesce(task.description, '') AS description
        ORDER BY task.name
        """,
        {},
    )
    return [Candidate(row["id"], row["name"], row["description"]) for row in rows]


def load_executable_strategies(
    url: str, password: str, task_id: str
) -> list[Candidate]:
    """Return only strategies whose entire graph path has local handlers."""
    rows = query_neo4j(
        url,
        password,
        """
        MATCH (:Task {id: $task_id})-[:USES]->(strategy:Strategy)
        RETURN strategy.id AS id, strategy.name AS name,
               coalesce(strategy.description, '') AS description
        ORDER BY strategy.name
        """,
        {"task_id": task_id},
    )
    candidates: list[Candidate] = []
    for row in rows:
        operations = load_operations(url, password, row["id"])
        if operations and all(OPERATIONS.contains(operation.id) for operation in operations):
            candidates.append(Candidate(row["id"], row["name"], row["description"]))
    return candidates


def select_candidate(
    client: DeepSeekClient,
    problem: Problem,
    kind: str,
    candidates: list[Candidate],
) -> tuple[Candidate, str, float | None]:
    """Let the model select exactly one ID from a supplied, finite candidate set."""
    if not candidates:
        raise LookupError(f"没有可执行的{kind}候选项")
    catalog = [candidate.__dict__ for candidate in candidates]
    result = client.select_id(
        system=(
            "你是数学题分类器。只从候选项中选择一个 id，不得创造 id。"
            "输出 JSON：selected_id（字符串）、reason（简短中文）、confidence（0 到 1 的数字）。"
        ),
        user=json.dumps(
            {
                "kind": kind,
                "problem": problem.__dict__,
                "candidates": catalog,
            },
            ensure_ascii=False,
        ),
        candidate_ids=[candidate.id for candidate in candidates],
    )
    selected_id = result.get("selected_id")
    selected = next((item for item in candidates if item.id == selected_id), None)
    if selected is None:
        raise ValueError(f"DeepSeek 返回了候选集之外的{kind} ID：{selected_id!r}")
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        confidence = None
    reason = result.get("reason")
    return selected, reason if isinstance(reason, str) else "", confidence


def match_task_from_question(
    problem: Problem,
    candidates: list[Candidate],
) -> Candidate | None:
    """Resolve an explicit task request locally; leave ambiguous text to the model."""
    text = problem.question_text.replace(" ", "")
    matched_ids = {
        task_id
        for task_id, keywords in TASK_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    }
    if len(matched_ids) != 1:
        return None
    matched_id = matched_ids.pop()
    return next((candidate for candidate in candidates if candidate.id == matched_id), None)


def classify_problem(
    problem: Problem,
    password: str,
    url: str,
    client: DeepSeekClient | None = None,
) -> Classification:
    """Choose a graph-valid Task, then a graph-valid executable Strategy."""
    client = client or DeepSeekClient()
    task_candidates = load_task_candidates(url, password)
    task = match_task_from_question(problem, task_candidates)
    if task is None:
        task, task_reason, task_confidence = select_candidate(
            client, problem, "Task", task_candidates
        )
    else:
        task_reason = "题干明确包含对应的题型关键词"
        task_confidence = 1.0

    strategy_candidates = load_executable_strategies(url, password, task.id)
    if len(strategy_candidates) == 1:
        strategy = strategy_candidates[0]
        strategy_reason = "该题型当前只有一个可执行策略"
        strategy_confidence = 1.0
    else:
        strategy, strategy_reason, strategy_confidence = select_candidate(
            client,
            problem,
            "Strategy",
            strategy_candidates,
        )
    reason = strategy_reason or task_reason
    confidence = strategy_confidence if strategy_confidence is not None else task_confidence
    return Classification(task, strategy, reason, confidence)
