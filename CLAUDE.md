# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**inspect-test-utils** is a utility library for the Inspect AI framework providing deterministic tasks, scorers, and a hardcoded model for integration tests, demos, and reproductions. It enables testing without external API calls.

## Commands

```bash
# Install dependencies
uv sync

# Linting and formatting
uv run ruff check              # Lint
uv run ruff format             # Format
uv run basedpyright            # Type check

# Testing
uv run pytest                  # All tests
uv run pytest tests/test_hardcoded.py::test_name  # Single test

# Run an evaluation
inspect eval inspect_test_utils/say_hello \
  --task-arg sample_count=3 \
  --model hardcoded --model-arg answer=hello
```

## Architecture

The library has 4 modules in `inspect_test_utils/`:

| Module | Purpose |
|--------|---------|
| `tasks.py` | Task definitions (`say_hello`, `guess_number`, `hardcoded_score`, `sometimes_fails_setup`, `sometimes_fails_scoring`, `configurable_sandbox`) |
| `scorers.py` | Scorer implementations (`failing_scorer`, `closeness_log`, `hardcoded_scorer`) |
| `hardcoded.py` | `HardcodedModelAPI` - deterministic model that emits pre-defined tool calls |
| `_registry.py` | Plugin registration for Inspect AI discovery |

**Plugin entry point:** Registered via `[project.entry-points.inspect_ai]` so tasks/models can be referenced directly in Inspect CLI.

## Key Patterns

- **Decorator registration:** Uses `@task`, `@scorer`, `@modelapi` decorators from Inspect AI
- **Parameterization:** All components accept initialization parameters, passable via `--task-arg` and `--model-arg` CLI flags
- **Error injection:** `failure_rate` and `fail_on_epochs` parameters for robustness testing
- **Tool call format:** `HardcodedToolCall` TypedDict with `tool_name` (str) and `tool_args` (dict)

## HardcodedModelAPI

The hardcoded model (`hardcoded.py`) emits a sequence of pre-defined tool calls, then submits a final answer:

```python
# Example: model that runs bash, then python, repeats twice, then answers
--model hardcoded \
--model-arg tool_calls='[{"tool_name": "bash", "tool_args": {"cmd": "echo hi"}}]' \
--model-arg repetitions=2 \
--model-arg answer="final answer"
```

Key parameters: `tool_calls`, `repetitions`, `answer`, `delay`, `concurrency`
