"""Execute Neo4j-defined quadratic strategies with trusted SymPy operations."""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import webbrowser
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import sympy as sp

from errors import ExecutionError, GraphServiceError
from operation_registry import OperationRegistry


TASK_ID = "quadratic-function-axis"
STRATEGY_ID = "axis-by-completing-square"
SAFE_EXPRESSION = re.compile(r"^[0-9xX+\-*/().\s]+$")
TEMPLATE_PATH = Path(__file__).with_name("templates") / "result.html"


@dataclass
class Operation:
    id: str
    name: str
    description: str
    display_description: str
    number: str
    ancestors: tuple[tuple[str, str], ...]
    condition_id: str | None


@dataclass
class Step:
    number: str
    name: str
    description: str
    display_description: str
    before: str
    after: str
    before_latex: str
    after_latex: str
    ancestors: tuple[tuple[str, str], ...]


@dataclass
class Answer:
    title: str
    text: str
    latex: str
    prefix: str = ""


@dataclass
class Solution:
    task_id: str
    task_name: str
    strategy_name: str
    expression: sp.Expr
    steps: list[Step]
    answer: Answer
    facts: dict[str, object]


def query_neo4j(url: str, password: str, cypher: str, parameters: dict) -> list[dict]:
    """Run a read-only Cypher query through Neo4j's HTTP transaction endpoint."""
    token = base64.b64encode(f"neo4j:{password}".encode()).decode()
    body = {"statements": [{"statement": cypher, "parameters": parameters}]}
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {token}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise GraphServiceError(f"知识图谱服务请求失败（HTTP {error.code}）。") from error
    except URLError as error:
        raise GraphServiceError("无法连接知识图谱服务，请确认 Neo4j 容器正在运行。") from error
    if payload["errors"]:
        raise GraphServiceError("知识图谱查询未被接受，请检查图谱配置。")
    result = payload["results"][0]
    return [dict(zip(result["columns"], item["row"])) for item in result["data"]]


def check_strategy(url: str, password: str, task_id: str, strategy_id: str) -> tuple[str, str]:
    rows = query_neo4j(
        url,
        password,
        """
        MATCH (task:Task {id: $task_id})-[:USES]->(strategy:Strategy {id: $strategy_id})
        RETURN task.name AS task_name, strategy.name AS strategy_name
        """,
        {"task_id": task_id, "strategy_id": strategy_id},
    )
    if not rows:
        raise LookupError("图谱中不存在该题型可选择的策略")
    return rows[0]["task_name"], rows[0]["strategy_name"]


def load_operations(url: str, password: str, strategy_id: str) -> list[Operation]:
    """Recursively expand Strategy -[:USES]-> Operation/Strategy."""
    seen: set[str] = set()

    def expand(
        current_id: str,
        prefix: tuple[int, ...] = (),
        ancestors: tuple[tuple[str, str], ...] = (),
    ) -> list[Operation]:
        if current_id in seen:
            raise ValueError("策略图中出现循环")
        seen.add(current_id)
        rows = query_neo4j(
            url,
            password,
            """
            MATCH (:Strategy {id: $strategy_id})-[r:USES]->(target)
            RETURN r.order AS order, labels(target) AS labels,
                   target.id AS id, target.name AS name,
                   target.description AS description,
                   coalesce(target.display_description, target.description) AS display_description,
                   r.condition_id AS condition_id
            ORDER BY order
            """,
            {"strategy_id": current_id},
        )
        result: list[Operation] = []
        for row in rows:
            number = ".".join(str(item) for item in (*prefix, row["order"]))
            if "Operation" in row["labels"]:
                result.append(
                    Operation(
                        row["id"],
                        row["name"],
                        row["description"],
                        row["display_description"],
                        number,
                        ancestors,
                        row["condition_id"],
                    )
                )
            elif "Strategy" in row["labels"]:
                result.extend(
                    expand(
                        row["id"],
                        (*prefix, row["order"]),
                        (*ancestors, (number, row["name"])),
                    )
                )
        seen.remove(current_id)
        return result

    return expand(strategy_id)


def parse_quadratic(text: str) -> tuple[sp.Symbol, sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Parse a trusted terminal expression and extract a, b, c."""
    if not SAFE_EXPRESSION.fullmatch(text):
        raise ValueError("MVP 只接受由 x、数字和 + - * / ( ) . 组成的 SymPy 表达式")
    x = sp.symbols("x", real=True)
    # The presentation/strategy expression retains the input's grouping.
    # Poly() below may still normalize a private copy to obtain a, b and c.
    expression = sp.sympify(
        text.replace("X", "x"), locals={"x": x}, evaluate=False
    )
    expression = normalize_presentation(expression)
    polynomial = sp.Poly(expression, x)
    if polynomial.degree() != 2:
        raise ValueError("MVP 目前只处理一元二次函数")
    a, b, c = polynomial.all_coeffs()
    if a == 0:
        raise ValueError("二次项系数必须非零")
    # Keep the input structure as the strategy's starting state. Poly() above
    # can analyse its coefficients without making the teaching trace start
    # with an implicit expansion.
    return x, expression, a, b, c


def normalize_presentation(expression: sp.Expr) -> sp.Expr:
    """Remove neutral arithmetic noise without expanding meaningful grouping."""
    if not expression.args:
        return expression

    arguments = tuple(normalize_presentation(argument) for argument in expression.args)
    if isinstance(expression, sp.Mul):
        numeric = sp.Mul(*(argument for argument in arguments if argument.is_number))
        symbolic = [argument for argument in arguments if not argument.is_number]
        if numeric == 0:
            return sp.Integer(0)
        factors = ([] if numeric == 1 else [numeric]) + symbolic
        if not factors:
            return sp.Integer(1)
        if len(factors) == 1:
            return factors[0]
        return sp.Mul(*factors, evaluate=False)

    if isinstance(expression, sp.Add):
        terms = [argument for argument in arguments if argument != 0]
        if not terms:
            return sp.Integer(0)
        if len(terms) == 1:
            return terms[0]
        return sp.Add(*terms, evaluate=False)

    if isinstance(expression, sp.Pow):
        base, exponent = arguments
        if exponent == 1:
            return base
        if exponent == 0:
            return sp.Integer(1)
        return sp.Pow(base, exponent, evaluate=False)

    return expression.func(*arguments)


def equation_text(right: sp.Expr) -> str:
    return f"y = {sp.sstr(right)}"


def equation_latex(right: sp.Expr) -> str:
    # Use the conventional descending-degree order for expanded polynomials.
    # It also leaves grouped factors, such as -1/2(x^2-8x), intact.
    return rf"y = {sp.latex(right, long_frac_ratio=0, order='lex')}"


def equivalent(left: sp.Expr, right: sp.Expr) -> bool:
    return sp.simplify(left - right) == 0


def with_quadratic_coefficient(a: sp.Expr, inner: sp.Expr, c: sp.Expr) -> sp.Expr:
    """Build a(inner)+c, omitting the conventional coefficient 1."""
    leading = inner if a == 1 else sp.Mul(a, inner, evaluate=False)
    return sp.Add(leading, c, evaluate=False)


def find_affine_square(original: sp.Expr, x: sp.Symbol) -> tuple[sp.Expr, sp.Expr, sp.Expr] | None:
    """Find a squared affine block mx+n in the original expression tree."""
    for power in original.atoms(sp.Pow):
        if power.exp != 2 or not isinstance(power.base, sp.Add):
            continue
        linear = sp.Poly(power.base, x)
        if linear.degree() == 1:
            m, n = linear.all_coeffs()
            if m != 0:
                return power.base, m, n
    return None


def condition_applies(condition_id: str | None, state: dict) -> bool:
    """Evaluate machine-readable conditions stored on Strategy USES edges."""
    if condition_id is None:
        return True
    if condition_id == "has-repeated-affine-term":
        return find_affine_square(state["source"], state["original_x"]) is not None
    if condition_id == "substitution-active":
        return "substitution" in state
    raise LookupError(f"未实现的图谱条件：{condition_id}")


OPERATIONS = OperationRegistry()


@OPERATIONS.register("op-whole-substitution")
def whole_substitution(state: dict) -> None:
    """Set u=mx+n, convert the function to a quadratic in u, and save x(u)."""
    x = state["x"]
    affine = find_affine_square(state["source"], x)
    if affine is None:
        raise ValueError("未找到可用于整体替换的线性整体")
    linear, m, n = affine
    u = sp.symbols("u", real=True)
    transformed = sp.expand(state["current"].subs(x, (u - n) / m))
    polynomial = sp.Poly(transformed, u)
    if polynomial.degree() != 2:
        raise ValueError("整体替换后不是关于 u 的一元二次函数")
    a, b, c = polynomial.all_coeffs()
    state.update(
        {
            "x": u,
            "a": a,
            "b": b,
            "c": c,
            "shift": sp.simplify(b / (2 * a)),
            "current": transformed,
            "substitution": {"m": m, "n": n},
            "after_text": f"令 u = {sp.sstr(linear)}，y = {sp.sstr(transformed)}",
            "after_latex": (
                rf"\begin{{aligned}}u &= {sp.latex(linear)} \\"
                rf"y &= {sp.latex(transformed, long_frac_ratio=0, order='lex')}\end{{aligned}}"
            ),
            # The displayed expression now uses u. Verify it by substituting
            # u = mx+n back into the original variable before comparing.
            "verify_rewrite": lambda before, after: equivalent(
                before, after.subs(u, linear)
            ),
        }
    )


@OPERATIONS.register("op-factor-quadratic-coefficient")
def factor_quadratic_coefficient(state: dict) -> None:
    x, a, b, c = state["x"], state["a"], state["b"], state["c"]
    state["current"] = with_quadratic_coefficient(
        a, sp.Add(x**2, sp.Mul(b / a, x, evaluate=False), evaluate=False), c
    )


@OPERATIONS.register("op-rewrite-linear-coefficient")
def rewrite_linear_coefficient(state: dict) -> None:
    x, a, c, shift = state["x"], state["a"], state["c"], state["shift"]
    state["current"] = with_quadratic_coefficient(
        a,
        sp.Add(x**2, sp.Mul(2, shift, x, evaluate=False), evaluate=False),
        c,
    )


@OPERATIONS.register("op-complete-square-term")
def complete_square_term(state: dict) -> None:
    x, a, c, shift = state["x"], state["a"], state["c"], state["shift"]
    state["current"] = with_quadratic_coefficient(
        a, sp.Add((x + shift) ** 2, -shift**2, evaluate=False), c
    )


@OPERATIONS.register("op-combine-constants")
def combine_constants(state: dict) -> None:
    x, a, b, c, shift = (
        state["x"],
        state["a"],
        state["b"],
        state["c"],
        state["shift"],
    )
    state["current"] = with_quadratic_coefficient(
        a, (x + shift) ** 2, sp.simplify(c - b**2 / (4 * a))
    )


@OPERATIONS.register("op-read-vertex-form")
def read_vertex_form(state: dict) -> None:
    variable = state["x"]
    vertex_x = sp.simplify(-state["b"] / (2 * state["a"]))
    vertex_y = sp.simplify(state["c"] - state["b"] ** 2 / (4 * state["a"]))
    state["axis"] = vertex_x
    state["vertex_value"] = vertex_y
    state["value_step"] = True
    state["value_after_text"] = (
        f"顶点为 ({sp.sstr(vertex_x)}, {sp.sstr(vertex_y)})"
    )
    state["value_after_latex"] = (
        rf"\left({sp.latex(vertex_x)},\,{sp.latex(vertex_y)}\right)"
    )


@OPERATIONS.register("op-calculate-axis-formula")
def calculate_axis_formula(state: dict) -> None:
    variable = state["x"]
    axis = sp.simplify(-state["b"] / (2 * state["a"]))
    state["axis"] = axis
    state["value_step"] = True
    state["value_after_text"] = f"{variable} = {sp.sstr(axis)}"
    state["value_after_latex"] = (
        rf"{sp.latex(variable)}=-\frac{{{sp.latex(state['b'])}}}"
        rf"{{2\cdot {sp.latex(state['a'])}}}={sp.latex(axis)}"
    )


@OPERATIONS.register("op-substitute-back-linear")
def substitute_back_linear(state: dict) -> None:
    """Solve the saved u=mx+n relation for the original variable x."""
    substitution = state["substitution"]
    u_axis = state["axis"]
    x_axis = sp.simplify((u_axis - substitution["n"]) / substitution["m"])
    state["axis"] = x_axis
    u = state["x"]
    original_x = state["original_x"]
    m, n = substitution["m"], substitution["n"]
    state["value_step"] = True
    state["value_before_text"] = (
        f"{sp.sstr(u)} = {sp.sstr(m * original_x + n)}，"
        f"{sp.sstr(u)} = {sp.sstr(u_axis)}"
    )
    state["value_before_latex"] = (
        rf"\begin{{aligned}}{sp.latex(u)} &= {sp.latex(m * original_x + n)} \\"
        rf"{sp.latex(u)} &= {sp.latex(u_axis)}\end{{aligned}}"
    )
    state["value_after_text"] = f"{sp.sstr(original_x)} = {sp.sstr(x_axis)}"
    state["value_after_latex"] = rf"{sp.latex(original_x)} = {sp.latex(x_axis)}"


@OPERATIONS.register("op-determine-extremum-by-sign")
def determine_extremum_by_sign(state: dict) -> None:
    coefficient = state["a"]
    if coefficient.is_positive:
        kind = "最小值"
    elif coefficient.is_negative:
        kind = "最大值"
    else:
        raise ValueError("无法根据二次项系数的符号判断最值类型")

    value = state.get("vertex_value")
    if value is None:
        raise LookupError("策略尚未读取顶点函数值")
    state["extremum_kind"] = kind
    state["extremum_value"] = value
    state["value_step"] = True
    state["value_after_text"] = f"函数取得{kind} {sp.sstr(value)}"
    state["value_after_latex"] = rf"\text{{{kind}}}={sp.latex(value)}"


def execute(operations: list[Operation], expression: sp.Expr, x: sp.Symbol,
            a: sp.Expr, b: sp.Expr, c: sp.Expr) -> tuple[list[Step], dict]:
    """Execute the operation IDs allowed by this MVP and verify each rewrite."""
    state = {
        "x": x,
        "a": a,
        "b": b,
        "c": c,
        "shift": sp.simplify(b / (2 * a)),
        "current": expression,
        "source": expression,
        "original_x": x,
    }
    steps: list[Step] = []

    for operation in operations:
        if not condition_applies(operation.condition_id, state):
            continue
        current = state["current"]
        before = equation_text(current)
        before_latex = equation_latex(current)
        state.pop("after_text", None)
        state.pop("after_latex", None)
        state.pop("verify_rewrite", None)
        state.pop("value_step", None)
        state.pop("value_before_text", None)
        state.pop("value_before_latex", None)
        state.pop("value_after_text", None)
        state.pop("value_after_latex", None)
        OPERATIONS.execute(operation.id, state)
        if state.get("value_step"):
            steps.append(
                Step(
                    operation.number,
                    operation.name,
                    operation.description,
                    operation.display_description,
                    state.get("value_before_text", before),
                    state["value_after_text"],
                    state.get("value_before_latex", before_latex),
                    state["value_after_latex"],
                    operation.ancestors,
                )
            )
            continue

        after = state["current"]
        verify_rewrite = state.get("verify_rewrite", equivalent)
        if not verify_rewrite(current, after):
            raise ArithmeticError(f"操作验证失败：{operation.name}")
        steps.append(
            Step(
                operation.number,
                operation.name,
                operation.description,
                operation.display_description,
                before,
                state.get("after_text", equation_text(after)),
                before_latex,
                state.get("after_latex", equation_latex(after)),
                operation.ancestors,
            )
        )
        current = after

    return steps, state


def build_answer(task_id: str, task_name: str, state: dict) -> Answer:
    if task_id == "quadratic-function-axis":
        if "axis" not in state:
            raise LookupError("策略路径没有产生对称轴")
        axis = state["axis"]
        return Answer(
            task_name,
            f"对称轴为 x = {sp.sstr(axis)}。",
            rf"x={sp.latex(axis)}",
            "对称轴为",
        )

    if task_id == "quadratic-function-vertex":
        required = {"axis", "vertex_value"}
        if not required.issubset(state):
            raise LookupError("策略路径没有产生完整的顶点信息")
        axis = state["axis"]
        value = state["vertex_value"]
        return Answer(
            task_name,
            f"顶点为 ({sp.sstr(axis)}, {sp.sstr(value)})。",
            rf"\left({sp.latex(axis)},\,{sp.latex(value)}\right)",
            "顶点为",
        )

    if task_id == "quadratic-function-extremum":
        required = {"axis", "extremum_kind", "extremum_value"}
        if not required.issubset(state):
            raise LookupError("策略路径没有产生完整的最值结果")
        axis = state["axis"]
        kind = state["extremum_kind"]
        value = state["extremum_value"]
        text = f"当 x = {sp.sstr(axis)} 时，函数取得{kind} {sp.sstr(value)}"
        latex = (
            rf"\text{{当 }}x={sp.latex(axis)}\text{{ 时，函数取得{kind} }}"
            rf"{sp.latex(value)}"
        )
        return Answer(task_name, text, latex, "")

    raise LookupError(f"尚未实现 Task 的答案生成器：{task_id}")


def collect_facts(state: dict) -> dict[str, object]:
    """Expose only verified reusable results, never the mutable execution state."""
    fact_names = ("axis", "vertex_value", "extremum_kind", "extremum_value")
    return {name: state[name] for name in fact_names if name in state}


def render_answer(answer: Answer) -> str:
    """Render one concise semantic answer without repeating it as display math."""
    formula = (
        f'{html.escape(answer.prefix)} \\({answer.latex}\\)'
        if answer.prefix
        else f'\\({answer.latex}\\)'
    )
    return "\n".join(["<h2>最终答案</h2>", f'<p class="answer">{formula}</p>'])


def render_solution_content(solution: Solution, heading_level: int = 1) -> str:
    """Render one verified task result without the surrounding HTML document."""
    expression = solution.expression
    strategy_name = solution.strategy_name
    steps = solution.steps
    answer = solution.answer
    heading = f"h{heading_level}"
    parts = [
        f"<{heading}>{html.escape(answer.title)}</{heading}>",
        f"<div class=\"formula\">\\({equation_latex(expression)}\\)</div>",
        f"<p>策略：{html.escape(strategy_name)}</p>",
    ]
    shown_strategies: set[tuple[str, str]] = set()
    for step in steps:
        for depth, strategy_node in enumerate(step.ancestors):
            if strategy_node not in shown_strategies:
                class_name = "strategy" if depth == 0 else "strategy nested"
                parts.append(
                    f'<h2 class="{class_name}">{html.escape(strategy_node[0])}. '
                    f"{html.escape(strategy_node[1])}</h2>"
                )
                shown_strategies.add(strategy_node)
        indent = " nested" if step.ancestors else ""
        parts.extend(
            [
                f'<section class="operation{indent}">',
                f"<h2>{html.escape(step.number)}. {html.escape(step.name)}</h2>",
                f'<div class="formula">\\({step.before_latex}\\)</div>',
                f'<p class="description">{html.escape(step.display_description)}</p>',
                f'<div class="formula">\\({step.after_latex}\\)</div>',
                "</section>",
            ]
        )
    parts.append(render_answer(answer))
    return "\n".join(parts)


def wrap_html(content: str, title: str = "数学解题过程") -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{{title}}", html.escape(title))
        .replace("{{content}}", content)
    )


def render_html(solution: Solution) -> str:
    """Render one verified task result as a complete MathJax page."""
    return wrap_html(render_solution_content(solution))


def solve_task(
    expression_text: str,
    password: str,
    task_id: str,
    strategy: str,
    url: str = "http://localhost:7474/db/math/tx/commit",
) -> Solution:
    """Run one graph-defined strategy and return structured verified output."""
    task_name, strategy_name = check_strategy(url, password, task_id, strategy)
    operations = load_operations(url, password, strategy)
    x, expression, a, b, c = parse_quadratic(expression_text)
    try:
        steps, state = execute(operations, expression, x, a, b, c)
        answer = build_answer(task_id, task_name, state)
    except (ArithmeticError, LookupError, ValueError) as error:
        raise ExecutionError(f"策略执行失败：{error}") from error
    return Solution(task_id, task_name, strategy_name, expression, steps, answer, collect_facts(state))


def solve_expression(
    expression_text: str,
    password: str,
    task_id: str = TASK_ID,
    strategy: str = STRATEGY_ID,
    url: str = "http://localhost:7474/db/math/tx/commit",
) -> str:
    """Run one graph-defined strategy and return its rendered result page."""
    return render_html(solve_task(expression_text, password, task_id, strategy, url))


def main() -> None:
    parser = argparse.ArgumentParser(description="Neo4j + SymPy：执行图谱定义的一元二次函数策略")
    parser.add_argument("--expression", required=True, help='例如 "x**2/2 - 5*x + 1"')
    parser.add_argument("--task", default=TASK_ID, help="Neo4j Task ID")
    parser.add_argument("--strategy", default=STRATEGY_ID, help="Neo4j Strategy ID")
    parser.add_argument("--url", default="http://localhost:7474/db/math/tx/commit")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("result.html")),
        help="输出的 HTML 文件",
    )
    parser.add_argument("--no-open", action="store_true", help="生成后不自动打开浏览器")
    args = parser.parse_args()

    password = os.getenv("NEO4J_PASSWORD") or getpass("Neo4j password: ")
    output = Path(args.output).resolve()
    output.write_text(
        solve_expression(args.expression, password, args.task, args.strategy, args.url), encoding="utf-8"
    )
    print(f"网页结果已写入：{output}")
    if not args.no_open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
