import math
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


@scorer(metrics=[accuracy(), stderr()])
def closeness_log() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer_str = state.output.completion
        try:
            a = float(answer_str)
        except ValueError:
            return Score(value=0.0)
        b = float(target.text)
        if a == b:
            return Score(value=1.0)
        rel = abs(a - b) / (abs(a) + abs(b))  # denominator is 0 only when a==b==0 (handled)
        return Score(value=1.0 / (1.0 + math.log1p(rel)))

    return score
