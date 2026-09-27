"""Placeholder test so CI has something to run at M0.

Replace/extend as each module lands: one test module per adapter, runner,
evaluator, and localization component (see the Testing & Reproducibility
section of the implementation plan).
"""

from llm_failure_localization import __version__


def test_version_is_set() -> None:
    assert __version__ == "0.0.1"
