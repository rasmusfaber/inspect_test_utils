"""Reusable testing framework for Inspect AI evaluation tasks.

This framework provides:
- Model-level mocking (HardcodedModelAPI)
- Solver-level hardcoded execution
- Test scorers and tasks
- Pytest fixtures and assertion helpers
- Eval runner for integration tests
"""

from inspect_test_utils.assertions import (
    assert_contains,
    assert_eval_score,
    assert_files_exist,
    assert_score_in_range,
)
from inspect_test_utils.eval_runner import EvalTestResult, run_eval_test, run_eval_test_async
from inspect_test_utils.solvers import (
    combined_solver,
    hardcoded_bash_solver,
    hardcoded_python_solver,
    inspection_solver,
)

__all__ = [
    "assert_contains",
    "assert_eval_score",
    "assert_files_exist",
    "assert_score_in_range",
    "combined_solver",
    "EvalTestResult",
    "hardcoded_bash_solver",
    "hardcoded_python_solver",
    "inspection_solver",
    "run_eval_test",
    "run_eval_test_async",
]
