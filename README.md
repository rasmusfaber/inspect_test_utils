inspect-test-utils

A small collection of tasks, scorers, and a simple model for use with the Inspect AI framework. It is designed to support integration/acceptance tests, demos, and reproductions by providing:
- Ready-made Tasks that exercise common evaluation patterns (simple generation, numeric closeness, failure injection, and sandbox configuration).
- Scorers for deterministic or parameterized scoring (including hardcoded outputs and a logarithmic closeness score).
- A hardcoded ModelAPI implementation that can deterministically emit tool calls and/or final answers, useful for testing tool-calling flows without hitting external APIs.

Quick start
- List tasks (via Inspect AI CLI):
  inspect ls tasks | grep inspect_test_utils
- Run a simple task with the hardcoded model:
  inspect eval inspect_test_utils/say_hello --model hardcoded --model-arg answer=hello

Passing arguments
Most tasks and the hardcoded model accept parameters. With the Inspect CLI you can pass them via --task-arg and --model-arg repeatedly:
- Example: make the task generate 3 samples and set a numeric target for guessing:
  inspect eval inspect_test_utils/guess_number \
    --task-arg sample_count=3 \
    --task-arg target=42.7 \
    --model hardcoded --model-arg answer=42.6

Python usage
You can also run tasks programmatically with Inspect AI's Python API.

Example: run "say_hello" with the hardcoded model

from inspect_ai import eval
from inspect_test_utils.tasks import say_hello
# The hardcoded model is registered as a modelapi called "hardcoded"
# You can select it via environment/CLI or your Inspect AI config.
result = eval(task=say_hello(sample_count=1), model="hardcoded")
print(result.summary())

Example: numeric closeness scoring (guess_number)

from inspect_ai import eval
from inspect_test_utils.tasks import guess_number
# The scorers.closeness_log() is used internally by this task
result = eval(task=guess_number(sample_count=2, target="42.7"), model="hardcoded")
print(result.metrics)

What’s included
- Tasks (inspect_test_utils.tasks)
  - say_hello(sample_count=1): Simple task; expects a response that includes "hello".
  - guess_number(sample_count=1, target="42.7"): Uses a logarithmic closeness scorer for numeric answers.
  - hardcoded_score(sample_count=10, hardcoded_score=None, hardcoded_score_by_sample_id_and_epoch=None): Scores are injected from parameters; useful for testing aggregations and edge cases (including NaN).
  - sometimes_fails_setup(sample_count=10, fail_setup_on_epochs=None, failure_rate=0.2): Randomly raises during setup via a failing solver; useful to test retry/resume behavior.
  - sometimes_fails_scoring(sample_count=10, fail_score_on_epochs=None, failure_rate=0.2): Randomly raises during scoring; useful to test scorer error handling.
  - configurable_sandbox(sample_count=1, cpu=0.5, memory="2G", storage="2G", gpu=None, gpu_model=None, allow_internet=False): Demonstrates configuring a Kubernetes sandbox by emitting a temporary values.yaml (supports T4/H100 selectors and internet egress opt-in).

- Scorers (inspect_test_utils.scorers)
  - failing_scorer(fail_on_epochs=None, failure_rate=0.2): Raises errors at a controlled rate for selected epochs.
  - closeness_log(): Scores 1.0 for exact equality, otherwise 1/(1+log1p(relative_error)) for numeric strings.
  - hardcoded_scorer(hardcoded_score=None, hardcoded_score_by_sample_id_and_epoch=None): Returns pre-specified Score objects or looks them up by sample id and epoch.

- Model (inspect_test_utils.hardcoded)
  - hardcoded: A ModelAPI that can emit a sequence of tool calls (e.g., bash) for a number of repetitions and then submit a final answer.
    Parameters include:
    - answer: final answer string (default: "done").
    - repetitions: how many tool-call "turns" before submitting.
    - tool_calls: list of tool calls or shell strings (e.g., ["echo hi", "ls -la"]) to simulate; defaults to none.
    - delay: optional async delay (seconds) before returning each model output.

Notes on usage
- CLI task selection uses the entry point namespace: inspect_test_utils/<task_name>.
- You can pass model parameters with --model-arg key=value and task parameters with --task-arg key=value.
- The "configurable_sandbox" task sets sandbox=("k8s", path_to_values_yaml) at runtime, which requires an Inspect AI setup that can launch Kubernetes jobs.

Development
- Set up dev environment:
  uv sync  # or: pip install -e ".[dev]"
- Lint / type-check / test:
  ruff check .
  basedpyright
  pytest -q

Troubleshooting
- If the CLI cannot find tasks, ensure the package is installed (pip install -e .) so entry points are discoverable.
- For sandboxed tasks, confirm your Inspect AI sandbox configuration (docker or k8s) is available.
- To observe injected failures, increase sample_count and set failure_rate close to 1.0 for reproducibility.

