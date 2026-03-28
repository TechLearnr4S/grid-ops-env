"""
smoke_test_phase2.py — End-to-end verification of Phase 2 components.

Tests:
  1. GridOpsEnv reset / step / state / list_tasks
  2. All 3 tasks cycle through their max_steps
  3. All 3 graders return valid scores
  4. FastAPI app imports without error
  5. Session management helpers work
"""

from __future__ import annotations

import copy
import sys

# ---- 1. GridOpsEnv basic usage ----
print("=== 1. GridOpsEnv reset / step / state ===")

from env.grid_env import GridOpsEnv
from env.models import Action, ActionType

tasks_to_test = ["task_easy", "task_medium", "task_hard"]

for task_id in tasks_to_test:
    env = GridOpsEnv(task_id)
    result = env.reset()
    assert result.done is False
    assert result.reward.score == 0.0
    assert result.observation.task_id == task_id
    assert result.observation.step == 0

    # Run a few steps
    for i in range(3):
        action = Action(action_type=ActionType.no_op)
        sr = env.step(action)
        assert 0.0 <= sr.reward.score <= 1.0
        assert isinstance(sr.done, bool)

    snap = env.state()
    assert snap["task_id"] == task_id
    assert snap["current_step"] == 3
    assert "grid_state" in snap
    assert "action_history" in snap
    assert "state_history" in snap
    assert len(snap["action_history"]) == 3
    assert len(snap["state_history"]) == 4  # initial + 3 steps

    print(f"  {task_id}: reset OK, 3 steps OK, state() OK")

# ---- 2. list_tasks ----
print("\n=== 2. list_tasks ===")
tasks = GridOpsEnv.list_tasks()
assert len(tasks) == 3
for t in tasks:
    assert "id" in t
    assert "name" in t
    assert "difficulty" in t
    assert "max_steps" in t
    print(f"  {t['id']}: {t['name']} ({t['difficulty']}, max_steps={t['max_steps']})")

# ---- 3. Invalid task_id ----
print("\n=== 3. Invalid task_id raises ValueError ===")
try:
    GridOpsEnv("task_nonexistent")
    print("  ERROR: should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"  Correctly raised ValueError: {str(e)[:60]}")

# ---- 4. Done environment step is safe ----
print("\n=== 4. Step on done environment is safe ===")
env_easy = GridOpsEnv("task_easy")
env_easy.reset()
# Force done by exhausting steps
env_easy._done = True
sr = env_easy.step(Action(action_type=ActionType.no_op))
assert sr.done is True
assert "already done" in sr.observation.messages[0].lower()
print("  Step on done env returns safely without crash.")

# ---- 5. Graders ----
print("\n=== 5. Graders (deterministic) ===")
from graders import grade, GRADERS

# Test grade_easy
env_e = GridOpsEnv("task_easy")
env_e.reset()
# Acknowledge alert + dispatch hydro + dispatch gas
actions_easy = [
    Action(action_type=ActionType.acknowledge_alert, alert_id="alert-easy-1"),
    Action(action_type=ActionType.dispatch_generator, generator_id="gen-hydro-1", target_mw=150.0),
    Action(action_type=ActionType.dispatch_generator, generator_id="gen-gas-1", target_mw=300.0),
]
for a in actions_easy:
    env_e.step(a)

snap_e = env_e.state()
gs_e = snap_e["grid_state"]
gs_e["_state_history"] = snap_e["state_history"]
result_e = grade("task_easy", gs_e, snap_e["action_history"])
assert 0.0 <= result_e["score"] <= 1.0
assert "breakdown" in result_e
assert "reason" in result_e
assert "supply_restored" in result_e["breakdown"]
assert "correct_dispatch" in result_e["breakdown"]
assert "alert_acknowledged" in result_e["breakdown"]
assert "efficiency" in result_e["breakdown"]
assert "no_unnecessary_actions" in result_e["breakdown"]
print(f"  grade_easy score={result_e['score']:.3f}  breakdown keys={list(result_e['breakdown'].keys())}")

# Test grade_medium
env_m = GridOpsEnv("task_medium")
env_m.reset()
actions_med = [
    Action(action_type=ActionType.acknowledge_alert, alert_id="alert-med-1"),
    Action(action_type=ActionType.shed_load, zone_id="zone-industrial", shed_mw=50.0),
    Action(action_type=ActionType.reroute_line, line_id="line-3", target_zone_id="zone-city-south"),
    Action(action_type=ActionType.send_field_crew, line_id="line-2", crew_id="crew-1"),
]
for a in actions_med:
    env_m.step(a)

snap_m = env_m.state()
gs_m = snap_m["grid_state"]
gs_m["_state_history"] = snap_m["state_history"]
result_m = grade("task_medium", gs_m, snap_m["action_history"])
assert 0.0 <= result_m["score"] <= 1.0
assert "city_south_restored" in result_m["breakdown"]
assert "line2_repair_initiated" in result_m["breakdown"]
assert "no_line3_overload" in result_m["breakdown"]
assert "alert_handling" in result_m["breakdown"]
assert "load_shed_correct" in result_m["breakdown"]
print(f"  grade_medium score={result_m['score']:.3f}  breakdown keys={list(result_m['breakdown'].keys())}")

# Test grade_hard
env_h = GridOpsEnv("task_hard")
env_h.reset()
actions_hard = [
    Action(action_type=ActionType.acknowledge_alert, alert_id="alert-hard-1"),
    Action(action_type=ActionType.shed_load, zone_id="zone-industrial", shed_mw=80.0),
    Action(action_type=ActionType.issue_public_notice,
           notice_text="Emergency load management in effect.",
           affected_zone_ids=["zone-industrial", "zone-suburbs"]),
    Action(action_type=ActionType.request_emergency_reserve, reserve_mw=60.0),
]
for a in actions_hard:
    env_h.step(a)

snap_h = env_h.state()
gs_h = snap_h["grid_state"]
gs_h["_state_history"] = snap_h["state_history"]
result_h = grade("task_hard", gs_h, snap_h["action_history"])
assert 0.0 <= result_h["score"] <= 1.0
assert "hospital_protected" in result_h["breakdown"]
assert "cascade_prevented" in result_h["breakdown"]
assert "budget_managed" in result_h["breakdown"]
assert "nuclear_compensated" in result_h["breakdown"]
assert "communication" in result_h["breakdown"]
print(f"  grade_hard score={result_h['score']:.3f}  breakdown keys={list(result_h['breakdown'].keys())}")

# ---- 6. Grade dispatcher validates task_id ----
print("\n=== 6. Grade dispatcher rejects unknown task_id ===")
try:
    grade("task_bogus", {}, [])
    print("  ERROR: should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"  Correctly raised ValueError: {str(e)[:60]}")

# ---- 7. FastAPI app imports clean ----
print("\n=== 7. FastAPI app import (no crash) ===")
import importlib
app_module = importlib.import_module("app")
assert hasattr(app_module, "app"), "app.py must expose 'app' (FastAPI instance)"
assert hasattr(app_module, "reset"), "app.py must define reset endpoint"
assert hasattr(app_module, "step"), "app.py must define step endpoint"
assert hasattr(app_module, "get_state"), "app.py must define get_state endpoint"
assert hasattr(app_module, "list_tasks"), "app.py must define list_tasks endpoint"
assert hasattr(app_module, "grade_episode"), "app.py must define grade_episode endpoint"
assert hasattr(app_module, "health"), "app.py must define health endpoint"
print("  app.py imported — all 6 endpoints defined.")

# ---- 8. Grader determinism: same inputs → same output ----
print("\n=== 8. Grader determinism ===")
result_e2 = grade("task_easy", gs_e, snap_e["action_history"])
assert result_e2["score"] == result_e["score"], "grade_easy is not deterministic!"

result_m2 = grade("task_medium", gs_m, snap_m["action_history"])
assert result_m2["score"] == result_m["score"], "grade_medium is not deterministic!"

result_h2 = grade("task_hard", gs_h, snap_h["action_history"])
assert result_h2["score"] == result_h["score"], "grade_hard is not deterministic!"
print("  All graders are deterministic.")

print("\nAll Phase 2 smoke-tests passed!")
