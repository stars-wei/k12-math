"""Detect every requested task, then execute only graph-backed capabilities."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from classifier import load_executable_strategies, load_task_candidates, select_candidate
from deepseek_client import DeepSeekClient
from problem import Candidate, Problem
from solve import (
    Answer,
    Solution,
    build_answer,
    render_answer,
    render_solution_content,
    solve_task,
    wrap_html,
)


@dataclass(frozen=True)
class TaskSpec:
    id: str
    name: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class TaskIntent:
    id: str
    name: str
    evidence: str


@dataclass
class TaskOutcome:
    intent: TaskIntent
    status: str
    solution: Solution | None = None
    answer: Answer | None = None
    detail: str = ""


@dataclass
class ProblemItemOutcome:
    label: str
    problem: Problem
    outcomes: list[TaskOutcome]


TASK_CATALOG = (
    TaskSpec(
        "quadratic-function-transformation",
        "判断一元二次函数图像的变换",
        (r"图像.{0,12}(?:怎样|如何|什么样的)?变换", r"平移", r"伸缩"),
    ),
    TaskSpec(
        "quadratic-function-vertex",
        "求一元二次函数图像的顶点",
        (r"顶点(?:坐标)?",),
    ),
    TaskSpec(
        "quadratic-function-axis",
        "求一元二次函数图像的对称轴",
        (r"对称轴",),
    ),
    TaskSpec(
        "quadratic-function-monotonicity",
        "判断一元二次函数的变化趋势",
        (r"变化趋势", r"单调性", r"单调递增", r"单调递减", r"增减性"),
    ),
    TaskSpec(
        "quadratic-function-extremum",
        "求一元二次函数最值",
        (r"最大值", r"最小值", r"最值", r"极大值", r"极小值", r"极值"),
    ),
)


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\\(", "").replace("\\)", ""))


def detect_task_intents(question: str) -> list[TaskIntent]:
    """Return all recognized requests in their first-occurrence order."""
    normalized = normalize_question(question)
    found: list[tuple[int, TaskIntent]] = []
    for spec in TASK_CATALOG:
        matches = [re.search(pattern, normalized) for pattern in spec.patterns]
        matches = [match for match in matches if match is not None]
        if not matches:
            continue
        first = min(matches, key=lambda match: match.start())
        found.append((first.start(), TaskIntent(spec.id, spec.name, first.group(0))))
    return [intent for _, intent in sorted(found, key=lambda item: item[0])]


def select_strategy(
    problem: Problem,
    task: Candidate,
    password: str,
    url: str,
    client: DeepSeekClient,
) -> Candidate | None:
    candidates = load_executable_strategies(url, password, task.id)
    if not candidates:
        return None
    requested = match_requested_strategy(problem.question_text, candidates)
    if requested is not None:
        return requested
    if len(candidates) == 1:
        return candidates[0]
    selected, _, _ = select_candidate(client, problem, "Strategy", candidates)
    return selected


def match_requested_strategy(
    question: str,
    candidates: list[Candidate],
) -> Candidate | None:
    """Honor an explicit method in the question before model-based selection."""
    normalized = normalize_question(question)
    method_names = ("配方法", "公式法", "对称法", "求导法", "顶点式读取法")
    requested_methods = [method for method in method_names if method in normalized]
    if len(requested_methods) != 1:
        return None
    method = requested_methods[0]
    matches = [candidate for candidate in candidates if method in candidate.name]
    return matches[0] if len(matches) == 1 else None


def solve_all_tasks(
    problem: Problem,
    password: str,
    url: str,
    client: DeepSeekClient,
) -> list[TaskOutcome]:
    intents = detect_task_intents(problem.question_text)
    if not intents:
        return [
            TaskOutcome(
                TaskIntent("unrecognized", "未识别题型", ""),
                "not_registered",
                detail="未能从题干中识别出已知题型",
            )
        ]

    graph_tasks = {task.id: task for task in load_task_candidates(url, password)}
    outcomes_by_position: dict[int, TaskOutcome] = {}
    facts: dict[str, object] = {}
    fact_sources: dict[str, str] = {}
    priority = {
        "quadratic-function-extremum": 0,
        "quadratic-function-vertex": 1,
        "quadratic-function-axis": 2,
    }
    execution_order = sorted(
        enumerate(intents), key=lambda item: (priority.get(item[1].id, 3), item[0])
    )
    for position, intent in execution_order:
        graph_task = graph_tasks.get(intent.id)
        if graph_task is None:
            outcomes_by_position[position] = TaskOutcome(
                intent,
                "not_registered",
                detail="知识图谱中尚没有该题型的可执行策略，本次未输出这部分答案。",
            )
            continue

        try:
            answer = build_answer(graph_task.id, graph_task.name, facts)
        except LookupError:
            answer = None
        if answer is not None:
            source_names = {
                fact_sources[name]
                for name in ("axis", "vertex_value", "extremum_kind", "extremum_value")
                if name in facts and name in fact_sources
            }
            source = "、".join(sorted(source_names)) or "前序计算"
            outcomes_by_position[position] = TaskOutcome(
                intent,
                "reused",
                answer=answer,
                detail=f"复用“{source}”已验证的中间结果。",
            )
            continue

        strategy = select_strategy(problem, graph_task, password, url, client)
        if strategy is None:
            outcomes_by_position[position] = TaskOutcome(
                intent,
                "not_registered",
                detail="知识图谱中尚没有该题型的可执行策略，本次未输出这部分答案。",
            )
            continue

        try:
            solution = solve_task(
                problem.expression_sympy,
                password,
                graph_task.id,
                strategy.id,
                url,
            )
        except Exception as error:
            outcomes_by_position[position] = TaskOutcome(intent, "failed", detail=str(error))
        else:
            facts.update(solution.facts)
            for fact_name in solution.facts:
                fact_sources.setdefault(fact_name, graph_task.name)
            outcomes_by_position[position] = TaskOutcome(intent, "solved", solution=solution)
    return [outcomes_by_position[index] for index in range(len(intents))]


def render_all_results(problem: Problem, outcomes: list[TaskOutcome]) -> str:
    return render_problem_items(
        problem.question_text,
        [ProblemItemOutcome("", problem, outcomes)],
    )


def render_outcome(outcome: TaskOutcome, heading_level: int) -> str:
    if outcome.status == "solved" and outcome.solution is not None:
        return render_solution_content(outcome.solution, heading_level=heading_level)
    if outcome.status == "reused" and outcome.answer is not None:
        heading = f"h{heading_level}"
        return (
            f"<{heading}>{html.escape(outcome.answer.title)}</{heading}>"
            f'<p class="reused">{html.escape(outcome.detail)}</p>'
            f"{render_answer(outcome.answer)}"
        )
    heading = f"h{heading_level}"
    if outcome.status == "not_registered":
        detail = outcome.detail or "知识图谱中尚没有该题型的可执行策略，本次未输出这部分答案。"
        return (
            f"<{heading}>{html.escape(outcome.intent.name)}</{heading}>"
            f'<p class="unsupported">{html.escape(detail)}</p>'
        )
    return (
        f"<{heading}>{html.escape(outcome.intent.name)}</{heading}>"
        '<p class="failed">题型已识别，但执行失败</p>'
        f'<p class="description">{html.escape(outcome.detail)}</p>'
    )


def render_problem_items(question: str, items: list[ProblemItemOutcome]) -> str:
    """Render one or more function items under their shared task requirements."""
    first_outcomes = items[0].outcomes if items else []
    recognized = "".join(
        f'<li><strong>{html.escape(outcome.intent.name)}</strong></li>'
        for outcome in first_outcomes
    )
    parts = [
        "<h1>题目分析与求解</h1>",
        f'<p class="question">{html.escape(question)}</p>',
        f'<p class="item-count">识别出 {len(items)} 个待求函数</p>',
        "<h2>识别出的题型</h2>",
        f'<ol class="recognized-tasks">{recognized}</ol>',
    ]
    multiple = len(items) > 1
    for item_index, item in enumerate(items, start=1):
        label = item.label.strip() or (f"（{item_index}）" if multiple else "")
        parts.append(f'<section class="problem-item" id="item-{item_index}">')
        if label:
            parts.append(f"<h2>{html.escape(label)}</h2>")
        for task_index, outcome in enumerate(item.outcomes, start=1):
            parts.append(
                f'<section class="task-result" id="item-{item_index}-task-{task_index}">'
            )
            parts.append(render_outcome(outcome, heading_level=3 if label else 2))
            parts.append("</section>")
        parts.append("</section>")
    return wrap_html("\n".join(parts), "题目分析与求解")
