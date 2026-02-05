from inspect_test_utils.hardcoded import hardcoded
from inspect_test_utils.scanners import suspicious_behaviour, word_counter
from inspect_test_utils.tasks import (
    configurable_sandbox,
    failing_solver,
    guess_number,
    guess_number_keep_guessing,
    hardcoded_score,
    network_sandbox,
    say_hello,
    say_hello_with_tools,
    sometimes_fails_scoring,
    sometimes_fails_setup,
    timeout,
)

__all__ = [
    "configurable_sandbox",
    "failing_solver",
    "guess_number",
    "guess_number_keep_guessing",
    "hardcoded",
    "hardcoded_score",
    "network_sandbox",
    "say_hello",
    "say_hello_with_tools",
    "sometimes_fails_scoring",
    "sometimes_fails_setup",
    "suspicious_behaviour",
    "timeout",
    "word_counter",
]
