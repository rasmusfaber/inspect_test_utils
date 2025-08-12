import random
from asyncio import sleep
from typing import Any, TypedDict, Callable

from inspect_ai import task, Task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ModelOutput,
    ChatCompletionChoice, modelapi, ModelAPI, ChatMessage, GenerateConfig, ModelCall,
)
from inspect_ai.scorer import includes, scorer, Target, Score, Scorer, accuracy, stderr
from inspect_ai.solver import solver, TaskState, Generate, use_tools, generate
from inspect_ai.tool import ToolCall, ToolInfo, ToolChoice, bash, python


@solver
def failing_solver(
        fail_on_epochs: list[int] | None = None,
        failure_rate: float = 0.2,
):
    async def solve(state: TaskState, generate: Generate):
        if fail_on_epochs is None or state.epoch in fail_on_epochs:
            if random.random() < failure_rate:
                raise ValueError("Eval failed!")

        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def failing_scorer(
        fail_on_epochs: list[int] | None = None,
        failure_rate: float = 0.2,
) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        if fail_on_epochs is None or state.epoch in fail_on_epochs:
            if random.random() < failure_rate:
                raise ValueError("Eval failed!")
        return Score(value=1.0)

    return score


@task
def sometimes_fails_setup(
        sample_count: int = 10,
        fail_setup_on_epochs: list[int] | None = None,
        failure_rate: float = 0.2,
) -> Task:
    return Task(
        dataset=[
            Sample(id=str(i), input="Say hello", target="hello") for i in range(sample_count)
        ],
        setup=failing_solver(fail_on_epochs=fail_setup_on_epochs, failure_rate=failure_rate),
        scorer=includes(),
        sandbox="docker",
        solver=[
            use_tools(bash(), python()),
            generate(),
        ]
    )


@task
def sometimes_fails_scoring(
        sample_count: int = 10,
        fail_score_on_epochs: list[int] | None = None,
        failure_rate: float = 0.2,
) -> Task:
    return Task(
        dataset=[
            Sample(id=str(i), input="Say hello", target="hello") for i in range(sample_count)
        ],
        scorer=failing_scorer(fail_on_epochs=fail_score_on_epochs, failure_rate=failure_rate),
        sandbox="docker",
        solver=[
            use_tools(bash(), python()),
            generate(),
        ]
    )
