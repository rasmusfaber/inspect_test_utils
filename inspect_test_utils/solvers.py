"""Hardcoded solvers for testing Inspect AI evaluations.

These solvers execute predetermined sequences of commands, useful for:
- Verifying that challenges are solvable with known solutions
- Testing scorer behavior with known outcomes
- Integration testing without model API calls
"""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox


def hardcoded_bash_solver(commands: list[str], timeout: int = 60) -> Solver:
    """Create a solver that executes a predefined sequence of bash commands.

    Args:
        commands: List of bash commands to execute in order.
        timeout: Timeout for each command in seconds.

    Returns:
        A solver that executes the commands sequentially.

    Example:
        solver = hardcoded_bash_solver([
            "sed -i 's/bug/fix/' /workspace/code.py",
            "pytest /workspace/tests -v",
        ])
    """

    @solver
    def solve() -> Solver:
        async def run(state: TaskState, generate: Generate) -> TaskState:
            for cmd in commands:
                await sandbox().exec(
                    ["bash", "-c", cmd],
                    timeout=timeout,
                )
            return state

        return run

    return solve()


def hardcoded_python_solver(code_blocks: list[str], timeout: int = 60) -> Solver:
    """Create a solver that executes predefined Python code blocks.

    Args:
        code_blocks: List of Python code strings to execute.
        timeout: Timeout for each code block in seconds.

    Returns:
        A solver that executes the code blocks sequentially.

    Example:
        solver = hardcoded_python_solver([
            '''
            import hashlib
            result = hashlib.md5(b"hello").hexdigest()
            print(result)
            '''
        ])
    """

    @solver
    def solve() -> Solver:
        async def run(state: TaskState, generate: Generate) -> TaskState:
            for code in code_blocks:
                await sandbox().exec(
                    ["python", "-c", code],
                    timeout=timeout,
                )
            return state

        return run

    return solve()


def inspection_solver(
    inspector: Callable[[TaskState], dict[str, Any] | Awaitable[dict[str, Any]]],
) -> Solver:
    """Create a solver that inspects sandbox state without modifying it.

    Useful for testing that the sandbox is set up correctly.

    Args:
        inspector: Function (sync or async) that receives TaskState and returns
            a dict of inspected values to store in state.metadata.

    Returns:
        A solver that runs the inspector and stores results.

    Example:
        async def check_files(state):
            result = await sandbox().exec(["ls", "/workspace"])
            return {"files": result.stdout.split()}

        solver = inspection_solver(check_files)
    """

    @solver
    def solve() -> Solver:
        async def run(state: TaskState, generate: Generate) -> TaskState:
            results = inspector(state)
            # Handle both sync and async inspectors
            if inspect.iscoroutine(results):
                results = await results
            if isinstance(results, dict):
                state.metadata["inspection_results"] = results
            return state

        return run

    return solve()


def combined_solver(*solvers: Solver) -> Solver:
    """Combine multiple solvers into a single solver chain.

    Args:
        *solvers: Solvers to run in sequence.

    Returns:
        A solver that runs all provided solvers in order.
    """

    @solver
    def solve() -> Solver:
        async def run(state: TaskState, generate: Generate) -> TaskState:
            for s in solvers:
                # Get the inner solve function
                inner = s
                if hasattr(s, "_solver"):
                    inner = s._solver  # type: ignore[union-attr]
                # Run the solver
                state = await inner(state, generate)  # type: ignore[misc]
            return state

        return run

    return solve()
