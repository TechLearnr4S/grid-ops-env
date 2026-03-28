"""
env/grid_env.py — Main environment loop for grid-ops-env.

GridOpsEnv wraps GridSimulator and task definitions into a clean
OpenEnv-compatible interface with reset / step / state / list_tasks.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .grid_simulator import GridSimulator
from .models import (
    Action,
    ActionType,
    GridState,
    Observation,
    Reward,
    StepResult,
)
from tasks.task_definitions import TASK_EASY, TASK_MEDIUM, TASK_HARD, get_task_by_id


# ---------------------------------------------------------------------------
# Registry of all available tasks (static metadata for list_tasks)
# ---------------------------------------------------------------------------

_ALL_TASKS: List[Dict[str, Any]] = [
    {
        "id": TASK_EASY["id"],
        "name": TASK_EASY["name"],
        "difficulty": TASK_EASY["difficulty"],
        "max_steps": TASK_EASY["max_steps"],
        "description": TASK_EASY["description"],
    },
    {
        "id": TASK_MEDIUM["id"],
        "name": TASK_MEDIUM["name"],
        "difficulty": TASK_MEDIUM["difficulty"],
        "max_steps": TASK_MEDIUM["max_steps"],
        "description": TASK_MEDIUM["description"],
    },
    {
        "id": TASK_HARD["id"],
        "name": TASK_HARD["name"],
        "difficulty": TASK_HARD["difficulty"],
        "max_steps": TASK_HARD["max_steps"],
        "description": TASK_HARD["description"],
    },
]

_VALID_TASK_IDS = {t["id"] for t in _ALL_TASKS}


class GridOpsEnv:
    """
    Power Grid Emergency Operations Center — OpenEnv environment.

    Usage:
        env = GridOpsEnv("task_easy")
        result = env.reset()
        result = env.step(Action(action_type=ActionType.no_op))
        snapshot = env.state()
    """

    def __init__(self, task_id: str) -> None:
        if task_id not in _VALID_TASK_IDS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {sorted(_VALID_TASK_IDS)}"
            )

        self._task_id: str = task_id
        self._task_config: Dict[str, Any] = get_task_by_id(task_id)
        self._max_steps: int = self._task_config["max_steps"]
        self._simulator: GridSimulator = GridSimulator()

        # Internal mutable state — initialised by reset()
        self._state: GridState = self._task_config["initial_state"].model_copy(deep=True)
        self._step: int = 0
        self._done: bool = False
        self._history: List[Dict[str, Any]] = []
        # Full per-step state snapshots (used by graders + /state endpoint)
        self._state_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> StepResult:
        """
        Reset the environment to its clean initial state.

        Fully deterministic — the same task always produces the same start.
        Returns a StepResult with a zero reward and done=False.
        """
        self._state = self._task_config["initial_state"].model_copy(deep=True)
        self._step = 0
        self._done = False
        self._history = []
        self._state_history = []

        # Record the initial state snapshot
        self._state_history.append(self._state.model_dump())

        observation = Observation(
            task_id=self._task_id,
            step=self._step,
            grid_state=self._state.model_copy(deep=True),
            messages=[
                f"Environment reset — task '{self._task_id}' ready. "
                f"Max steps: {self._max_steps}."
            ],
            done=False,
        )
        reward = Reward(
            score=0.0,
            breakdown={},
            reason="Episode start — no reward on reset.",
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=False,
            info={
                "task_id": self._task_id,
                "max_steps": self._max_steps,
                "event": "reset",
            },
        )

    def step(self, action: Action) -> StepResult:
        """
        Execute one action in the environment.

        Pipeline:
          1. Validate action (invalid type → penalty reward, no crash)
          2. apply_action  → new_state, reward_parts, messages
          3. tick          → advances time, checks cascades
          4. calculate_reward → Reward
          5. check_done
          6. Record history, return StepResult
        """
        if self._done:
            # Episode already finished — return terminal observation
            observation = Observation(
                task_id=self._task_id,
                step=self._step,
                grid_state=self._state.model_copy(deep=True),
                messages=["Episode is already done. Call reset() to start a new episode."],
                done=True,
            )
            return StepResult(
                observation=observation,
                reward=Reward(score=0.0, breakdown={}, reason="Episode already done."),
                done=True,
                info={"warning": "step called on a done environment"},
            )

        prev_state = self._state.model_copy(deep=True)

        # -- 1. Validate action type --
        if not isinstance(action.action_type, ActionType):
            penalty_obs = Observation(
                task_id=self._task_id,
                step=self._step,
                grid_state=self._state.model_copy(deep=True),
                messages=[f"Invalid action_type '{action.action_type}' — 0.0 penalty applied."],
                done=False,
            )
            return StepResult(
                observation=penalty_obs,
                reward=Reward(
                    score=0.0,
                    breakdown={"invalid_action_penalty": -0.1},
                    reason="Invalid action type submitted.",
                ),
                done=False,
                info={"error": "invalid_action_type"},
            )

        # -- 2. Apply action --
        new_state, _parts, action_messages = self._simulator.apply_action(
            self._state, action
        )

        # -- 3. Tick (advance time, resolve cascades) --
        new_state, tick_events = self._simulator.tick(new_state)
        all_messages = action_messages + tick_events

        # -- 4. Compute reward --
        reward = self._simulator.calculate_reward(
            new_state, prev_state, action, all_messages
        )

        # -- 5. Check done --
        self._step += 1
        new_state.current_step = self._step

        done = self._simulator.check_done(new_state, self._max_steps)
        if not done:
            # Additional done: all zones are fully served (victory condition)
            if new_state.total_mw_demanded > 0:
                if new_state.total_mw_served >= new_state.total_mw_demanded * 0.98:
                    from env.models import LoadZone
                    if all(not z.is_blacked_out for z in new_state.load_zones):
                        done = True
                        all_messages.append(
                            "🎉 All load zones served — episode complete (victory)!"
                        )

        self._done = done
        self._state = new_state

        # -- 6. Record history --
        action_record = action.model_dump()
        action_record["_step"] = self._step
        self._history.append(action_record)
        self._state_history.append(new_state.model_dump())

        observation = Observation(
            task_id=self._task_id,
            step=self._step,
            grid_state=self._state.model_copy(deep=True),
            messages=all_messages,
            done=done,
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info={
                "step": self._step,
                "max_steps": self._max_steps,
                "budget_remaining": new_state.budget_remaining,
                "mw_served": new_state.total_mw_served,
                "mw_demanded": new_state.total_mw_demanded,
                "active_blackouts": sum(
                    1 for z in new_state.load_zones if z.is_blacked_out
                ),
            },
        )

    def state(self) -> Dict[str, Any]:
        """
        Return a full JSON-serialisable snapshot of the current environment state.

        Keys:
          task_id, current_step, done, grid_state, action_history, state_history
        """
        return {
            "task_id": self._task_id,
            "current_step": self._step,
            "max_steps": self._max_steps,
            "done": self._done,
            "grid_state": self._state.model_dump(),
            "action_history": list(self._history),
            # Full per-step snapshots — needed by graders (esp. grade_hard hospital check)
            "state_history": list(self._state_history),
        }

    @staticmethod
    def list_tasks() -> List[Dict[str, Any]]:
        """Return metadata for all available tasks."""
        return list(_ALL_TASKS)
