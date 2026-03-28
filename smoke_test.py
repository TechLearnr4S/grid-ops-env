"""Smoke test: verifies all task definitions and all 8 action types work end-to-end."""
import copy
import sys

from tasks.task_definitions import TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.grid_simulator import GridSimulator
from env.models import Action, ActionType

sim = GridSimulator()

# Demand curve sanity
print("--- Demand Curve ---")
for h in [3, 4, 5, 12, 16, 18, 20]:
    print(f"  hour {h:02d}: {sim._demand_curve(h):.3f}x")

assert sim._demand_curve(4) <= sim._demand_curve(12), "Hour 4 should be less than hour 12"
assert sim._demand_curve(4) <= sim._demand_curve(16), "Hour 4 should be the trough"
assert sim._demand_curve(16) >= sim._demand_curve(18), "Hour 16 is the peak"

# Task loading
for task in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
    state = task["initial_state"]
    print(f"\n--- {task['id']} ---")
    print(f"  generators={len(state.generators)}, lines={len(state.lines)}, "
          f"zones={len(state.load_zones)}, alerts={len(state.alerts)}")
    print(f"  demand={state.total_mw_demanded:.0f} MW, served={state.total_mw_served:.0f} MW, "
          f"budget=${state.budget_remaining:.0f}")
    assert state.total_mw_served <= state.total_mw_demanded + 1e-6

# All 8 action types smoke test on TASK_EASY
state = copy.deepcopy(TASK_EASY["initial_state"])
prev_state = copy.deepcopy(state)

print("\n--- Action smoke-test on TASK_EASY (all 8 types) ---")
actions = [
    Action(action_type=ActionType.no_op),
    Action(action_type=ActionType.acknowledge_alert, alert_id="alert-easy-1"),
    Action(action_type=ActionType.dispatch_generator, generator_id="gen-hydro-1", target_mw=120.0),
    Action(action_type=ActionType.dispatch_generator, generator_id="gen-gas-1", target_mw=300.0),
    Action(action_type=ActionType.shed_load, zone_id="zone-industrial", shed_mw=20.0),
    Action(action_type=ActionType.request_emergency_reserve, reserve_mw=30.0),
    Action(action_type=ActionType.issue_public_notice,
           notice_text="Temporary load management in effect.",
           affected_zone_ids=["zone-industrial"]),
    Action(action_type=ActionType.send_field_crew, line_id="line-A", crew_id="crew-1"),
]

for action in actions:
    state2, parts, msgs = sim.apply_action(state, action)
    reward = sim.calculate_reward(state2, prev_state, action, msgs)
    label = msgs[0][:55] if msgs else "(no message)"
    print(f"  {action.action_type.value:35s}: score={reward.score:.3f}  | {label}")
    assert 0.0 <= reward.score <= 1.0, f"Score out of range: {reward.score}"
    prev_state = state
    state = state2

# Test reroute separately on TASK_MEDIUM (which has a faulted line to avoid)
state_m = copy.deepcopy(TASK_MEDIUM["initial_state"])
reroute = Action(action_type=ActionType.reroute_line, line_id="line-3", target_zone_id="zone-city-south")
state_m2, _, msgs_m = sim.apply_action(state_m, reroute)
print(f"\n  reroute_line (MEDIUM task)        : {msgs_m[0][:55] if msgs_m else '(no message)'}")

# Tick
state, events = sim.tick(state)
print(f"\nAfter tick: step={state.current_step}, hour={state.current_hour}, "
      f"demand={state.total_mw_demanded:.1f}, served={state.total_mw_served:.1f}")

done = sim.check_done(state, TASK_EASY["max_steps"])
print(f"Done: {done}")

print("\n✅  All smoke-tests passed!")
