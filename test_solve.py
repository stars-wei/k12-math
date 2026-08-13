"""Unit tests for trusted quadratic operations; no Neo4j or API is required."""

from __future__ import annotations

import unittest

import sympy as sp

from solve import Operation, build_answer, equation_latex, execute, parse_quadratic


def operation(
    operation_id: str,
    number: str,
    condition_id: str | None = None,
) -> Operation:
    return Operation(operation_id, operation_id, "", "", number, (), condition_id)


BASE_OPERATIONS = [
    operation("op-whole-substitution", "0", "has-repeated-affine-term"),
    operation("op-factor-quadratic-coefficient", "1"),
    operation("op-rewrite-linear-coefficient", "2"),
    operation("op-complete-square-term", "3"),
    operation("op-combine-constants", "4"),
    operation("op-read-vertex-form", "5"),
    operation("op-substitute-back-linear", "6", "substitution-active"),
]


class QuadraticSolverTests(unittest.TestCase):
    def execute_expression(self, text: str, extremum: bool = False) -> dict:
        x, expression, a, b, c = parse_quadratic(text)
        operations = list(BASE_OPERATIONS)
        if extremum:
            operations.append(operation("op-determine-extremum-by-sign", "7"))
        _, state = execute(operations, expression, x, a, b, c)
        return state

    def test_axis(self) -> None:
        state = self.execute_expression("x**2/2 - 5*x + 1")
        answer = build_answer("quadratic-function-axis", "求对称轴", state)
        self.assertEqual(state["axis"], sp.Integer(5))
        self.assertEqual(answer.text, "x = 5")

    def test_axis_formula(self) -> None:
        x, expression, a, b, c = parse_quadratic("x**2/2 - 5*x + 1")
        _, state = execute(
            [operation("op-calculate-axis-formula", "1")],
            expression,
            x,
            a,
            b,
            c,
        )
        self.assertEqual(state["axis"], sp.Integer(5))

    def test_redundant_unit_factors_are_not_displayed(self) -> None:
        _, expression, _, _, _ = parse_quadratic("1*1/2*1*x**2 + 2*x + 5")
        self.assertEqual(sp.sstr(expression), "x**2/2 + 2*x + 5")
        self.assertEqual(equation_latex(expression), r"y = \frac{1}{2} x^{2} + 2 x + 5")

    def test_grouped_affine_expression_is_not_expanded(self) -> None:
        _, expression, _, _, _ = parse_quadratic("(2*x-1)**2 + 6*(2*x-1) + 5")
        rendered = sp.sstr(expression)
        self.assertIn("(2*x - 1)**2", rendered)
        self.assertIn("6*(2*x - 1)", rendered)

    def test_minimum(self) -> None:
        state = self.execute_expression("x**2 - 4*x + 3", extremum=True)
        answer = build_answer("quadratic-function-extremum", "求最值", state)
        self.assertEqual(state["axis"], sp.Integer(2))
        self.assertEqual(state["extremum_value"], sp.Integer(-1))
        self.assertEqual(state["extremum_kind"], "最小值")
        self.assertIn("最小值 -1", answer.text)

    def test_maximum(self) -> None:
        state = self.execute_expression("-x**2 + 4*x + 1", extremum=True)
        self.assertEqual(state["axis"], sp.Integer(2))
        self.assertEqual(state["extremum_value"], sp.Integer(5))
        self.assertEqual(state["extremum_kind"], "最大值")

    def test_affine_substitution_extremum(self) -> None:
        state = self.execute_expression("(2*x-1)**2 + 6*(2*x-1) + 5", extremum=True)
        self.assertEqual(state["axis"], sp.Integer(-1))
        self.assertEqual(state["extremum_value"], sp.Integer(-4))
        self.assertEqual(state["extremum_kind"], "最小值")


if __name__ == "__main__":
    unittest.main()
