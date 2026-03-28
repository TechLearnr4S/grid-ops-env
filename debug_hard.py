import sys, traceback, copy

from env.models import Action, ActionType, GridState, LoadZone, ZonePriority
from env.grid_simulator import GridSimulator
from tasks.task_definitions import TASK_HARD

state = TASK_HARD["initial_state"].model_copy(deep=True)
sim = GridSimulator()

print("Initial:")
print(f"  demand={state.total_mw_demanded:.1f}, served={state.total_mw_served:.1f}")
for ln in state.lines:
    print(f"  line {ln.id}: current={ln.current_mw}, max={ln.max_mw}, status={ln.status}")

action = Action(action_type=ActionType.no_op)
new_state, _, msgs = sim.apply_action(state, action)

print("\nAfter apply_action:")
for ln in new_state.lines:
    print(f"  line {ln.id}: current={ln.current_mw}, max={ln.max_mw}, status={ln.status}")
print(f"  demand={new_state.total_mw_demanded:.1f}, served={new_state.total_mw_served:.1f}")

# Manually replicate the cascade step without model_copy to see what fails
print("\nSimulating cascade manually...")
for ln in new_state.lines:
    if ln.status.value == "overloaded":
        if ln.current_mw > ln.max_mw:
            print(f"  {ln.id} will trip -> faulted, current_mw -> 0")
            target = next((z for z in new_state.load_zones if z.id == ln.to_zone), None)
            if target:
                lost = min(target.served_mw, ln.current_mw)
                print(f"  zone {target.id}: served {target.served_mw} -> {target.served_mw - lost}")
                target.served_mw = max(0.0, target.served_mw - lost)
            ln.status_val = "faulted"
            ln.current_mw = 0.0

print("\nZones after cascade:")
for z in new_state.load_zones:
    print(f"  {z.id}: demand={z.demand_mw:.1f}, served={z.served_mw:.1f}")
total_demanded = sum(z.demand_mw for z in new_state.load_zones)
total_served = sum(z.served_mw for z in new_state.load_zones)
print(f"  TOTALS: demanded={total_demanded:.1f}, served={total_served:.1f}")

print("\nNow trying tick()...")
try:
    ticked, events = sim.tick(new_state)
    print("tick() OK")
except Exception as e:
    print(f"tick() FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
