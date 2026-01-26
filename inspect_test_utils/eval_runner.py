"""Eval runner helpers for testing Inspect AI evaluations.

Provides a simplified interface for running evals in tests with custom solvers.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from inspect_ai import Task, eval
from inspect_ai.log import EvalLog
from inspect_ai.solver import Solver


@dataclass
class EvalTestResult:
    """Result of running an eval test.

    Attributes:
        success: Whether the eval completed without errors.
        score: The score from the first sample (if available).
        scores: All scores from the eval (for multi-sample tests).
        explanation: Score explanation (if available).
        metadata: Score metadata (if available).
        log: The full EvalLog for detailed inspection.
        error: Error message if the eval failed.
    """

    success: bool
    score: float | None = None
    scores: list[float] = field(default_factory=list)
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    log: EvalLog | None = None
    error: str | None = None

    @classmethod
    def from_log(cls, log: EvalLog) -> "EvalTestResult":
        """Create an EvalTestResult from an EvalLog."""
        if log.status != "success":
            return cls(
                success=False,
                log=log,
                error=f"Eval status: {log.status}. Error: {log.error}",
            )

        # Extract scores from samples
        scores: list[float] = []
        explanation: str | None = None
        metadata: dict[str, Any] = {}

        if log.samples:
            for sample in log.samples:
                if sample.scores:
                    for score_name, score in sample.scores.items():
                        if score.value is not None:
                            if isinstance(score.value, (int, float)):
                                scores.append(float(score.value))
                            # Capture first explanation/metadata for convenience
                            if explanation is None:
                                explanation = score.explanation
                            if not metadata:
                                metadata = score.metadata or {}

        return cls(
            success=True,
            score=scores[0] if scores else None,
            scores=scores,
            explanation=explanation,
            metadata=metadata,
            log=log,
        )


def run_eval_test(
    task: Task | Callable[[], Task],
    solver: Solver | list[Solver] | None = None,
    *,
    limit: int | None = 1,
    model: str = "mockllm/model",
    sandbox_cleanup: bool = True,
    max_messages: int | None = None,
    **eval_kwargs: Any,
) -> EvalTestResult:
    """Run an eval for testing and return a simplified result.

    This is the main entry point for running eval tests. It:
    - Runs the eval with the provided solver
    - Uses mockllm by default (no API calls)
    - Returns a simplified result for assertions

    Args:
        task: The Task or task function to evaluate.
        solver: Custom solver(s) to use. If None, uses the task's default solver.
        limit: Number of samples to run (default 1 for fast tests).
        model: Model to use (default mockllm/model for no API calls).
        sandbox_cleanup: Whether to cleanup sandbox after test (default True).
        max_messages: Override max_messages for the task.
        **eval_kwargs: Additional kwargs passed to eval().

    Returns:
        EvalTestResult with score and metadata.

    Example:
        from inspect_eval_testing import run_eval_test, hardcoded_bash_solver

        def test_code_repair():
            result = run_eval_test(
                code_repair,
                solver=hardcoded_bash_solver([
                    "sed -i 's/bug/fix/' /workspace/code.py",
                ]),
                expected_score=1.0,
            )
            assert result.score == 1.0
    """
    # Get the task if it's a callable
    task_instance = task() if callable(task) else task

    # Override max_messages if provided
    if max_messages is not None:
        task_instance = Task(
            dataset=task_instance.dataset,
            solver=task_instance.solver if solver is None else solver,
            scorer=task_instance.scorer,
            max_messages=max_messages,
            sandbox=task_instance.sandbox,
            metadata=task_instance.metadata,
        )
    elif solver is not None:
        task_instance = Task(
            dataset=task_instance.dataset,
            solver=solver,
            scorer=task_instance.scorer,
            max_messages=task_instance.max_messages,
            sandbox=task_instance.sandbox,
            metadata=task_instance.metadata,
        )

    try:
        logs = eval(
            task_instance,
            model=model,
            limit=limit,
            sandbox_cleanup=sandbox_cleanup,
            **eval_kwargs,
        )

        if not logs:
            return EvalTestResult(success=False, error="No eval logs returned")

        return EvalTestResult.from_log(logs[0])

    except Exception as e:
        return EvalTestResult(success=False, error=str(e))


def run_eval_test_async(
    task: Task | Callable[[], Task],
    solver: Solver | list[Solver] | None = None,
    *,
    limit: int | None = 1,
    model: str = "mockllm/model",
    sandbox_cleanup: bool = True,
    **eval_kwargs: Any,
) -> EvalTestResult:
    """Async version of run_eval_test.

    Same as run_eval_test but uses eval_async internally.
    Use this in async test functions.
    """
    import asyncio

    from inspect_ai import eval_async

    async def _run() -> EvalTestResult:
        task_instance = task() if callable(task) else task

        if solver is not None:
            task_instance = Task(
                dataset=task_instance.dataset,
                solver=solver,
                scorer=task_instance.scorer,
                max_messages=task_instance.max_messages,
                sandbox=task_instance.sandbox,
                metadata=task_instance.metadata,
            )

        try:
            logs = await eval_async(
                task_instance,
                model=model,
                limit=limit,
                sandbox_cleanup=sandbox_cleanup,
                **eval_kwargs,
            )

            if not logs:
                return EvalTestResult(success=False, error="No eval logs returned")

            return EvalTestResult.from_log(logs[0])

        except Exception as e:
            return EvalTestResult(success=False, error=str(e))

    return asyncio.get_event_loop().run_until_complete(_run())
