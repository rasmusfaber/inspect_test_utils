import random
from asyncio import sleep
from typing import Any, TypedDict, override

import inspect_ai._util.constants
import yaml
from inspect_ai.model import (
    ChatMessageAssistant,
    ModelOutput,
    ChatCompletionChoice, modelapi, ModelAPI, ChatMessage, GenerateConfig, ModelCall,
)
from inspect_ai.tool import ToolCall, ToolInfo, ToolChoice


class HardcodedToolCall(TypedDict):
    tool_name: str
    tool_args: dict[str, Any]


class HardcodedModelAuthFailure(Exception):
    pass


class HardcodedModelAPI(ModelAPI):
    def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config: GenerateConfig = GenerateConfig(),
            tool_calls: list[HardcodedToolCall] | str | list[str] | None = None,
            tool_call_file=None,
            repetitions: int = 1,
            answer: str = "done",
            delay: float = 0.0,
            concurrency: int = inspect_ai._util.constants.DEFAULT_MAX_CONNECTIONS,
            auth_failure_chance: float = 0.0,
    ):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, api_key_vars=["ANTHROPIC_API_KEY"], config=config)
        self.tool_calls = self._parse_tool_calls(tool_calls) or self._parse_tool_call_file(tool_call_file)
        self.repetitions = repetitions
        self.answer = answer
        self.delay = delay
        self.concurrency = concurrency
        self.auth_failure_chance = auth_failure_chance



    def _parse_tool_calls(self, tool_calls: list[HardcodedToolCall] | str | list[str] | None) -> list[
        HardcodedToolCall]:
        if tool_calls is None:
            return []
        if isinstance(tool_calls, str):
            tool_calls = [tool_calls]
        if len(tool_calls) == 0:
            return []
        if isinstance(tool_calls[0], str):
            return [HardcodedToolCall(tool_name='bash', tool_args={'cmd': cmd}) for cmd in tool_calls]
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise ValueError(f"Invalid tool call: {tool_call}")
            if "tool_name" not in tool_call or "tool_args" not in tool_call:
                raise ValueError(f"Invalid tool call: {tool_call}")
        return tool_calls

    def _parse_tool_call_file(self, tool_call_file: str | None) -> list[HardcodedToolCall]:
        if tool_call_file is None:
            return []
        with open(tool_call_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def max_connections(self) -> int:
        return self.concurrency

    @override
    async def generate(
            self,
            input: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        if self.auth_failure_chance > 0 and random.random() < self.auth_failure_chance:
            raise HardcodedModelAuthFailure()
        index = (len(input) - 1) // 2
        next_tool_call_index = int(index) % len(self.tool_calls) if self.tool_calls else 0
        repetition_count = int(index) // len(self.tool_calls) if self.tool_calls else 1
        next_tool_call = self.tool_calls[next_tool_call_index] if next_tool_call_index < len(self.tool_calls) else None
        if self.delay > 0:
            await sleep(self.delay)

        if repetition_count >= self.repetitions:
            submit_tool = next((tool for tool in tools if tool.name == "submit"), None)
            if submit_tool is None:
                message = ChatMessageAssistant(content=self.answer)
            else:
                message = ChatMessageAssistant(
                    content="I will now submit my answer.",
                    tool_calls=[
                        ToolCall(
                            id="hardcoded_submit",
                            function=submit_tool.name,
                            arguments={"answer": self.answer},
                        )
                    ],
                )

            choice = ChatCompletionChoice(
                message=message,
                stop_reason="stop",
            )
        else:
            tool_name = next_tool_call["tool_name"]
            tool_args = next_tool_call["tool_args"]

            message = ChatMessageAssistant(
                content=f"Executing {tool_name} with args: {tool_args}",
                tool_calls=[
                    ToolCall(
                        id=f"hardcoded_{index}",
                        function=tool_name,
                        arguments=tool_args,
                    )
                ],
            )
            choice = ChatCompletionChoice(message=message)

        return ModelOutput(
            model="hardcoded", choices=[choice]
        )

    @override
    def should_retry(self, ex: Exception) -> bool:
        return isinstance(ex, HardcodedModelAuthFailure)

    @override
    def is_auth_failure(self, ex: Exception) -> bool:
        return isinstance(ex, HardcodedModelAuthFailure)


@modelapi(name="hardcoded")
def hardcoded() -> type[ModelAPI]:
    return HardcodedModelAPI
