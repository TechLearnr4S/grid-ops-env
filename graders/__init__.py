"""
graders/__init__.py — Grader registry for grid-ops-env.

Usage:
    from graders import grade
    result = grade("task_easy", final_state, action_history)
"""

from __future__ import annotations

from typing import Any, Dict, List

from .grade_easy import grade_easy
from .grade_medium import grade_medium
from .grade_hard import grade_hard

GRADERS = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}


def grade(task_id: str, final_state: dict, action_history: List[Dict[str, Any]]) -> dict:
    """
    Dispatch to the appropriate grader for the given task_id.

    Args:
        task_id:        One of "task_easy", "task_medium", "task_hard"
        final_state:    env.state()["grid_state"]  (dict form of GridState)
        action_history: env.state()["action_history"]

    Returns:
        {"score": float, "breakdown": dict, "reason": str}

    Raises:
        ValueError: if task_id is not recognised
    """
    if task_id not in GRADERS:
        raise ValueError(
            f"Unknown task_id: '{task_id}'. "
            f"Available: {sorted(GRADERS.keys())}"
        )
    return GRADERS[task_id](final_state, action_history)


__all__ = [
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "GRADERS",
    "grade",
]
