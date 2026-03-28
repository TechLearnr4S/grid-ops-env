"""
graders/grade_medium.py — Deterministic grader for TASK_MEDIUM "Line Fault Recovery".

Criteria and weights:
  city_south_restored   (0.30) — City South (zone-city-south) not blacked out in final state
  line2_repair_initiated(0.20) — history contains send_field_crew targeting line-2
  no_line3_overload     (0.20) — Line3 never exceeded 100% capacity in any step snapshot
  alert_handling        (0.15) — critical alert acknowledged within first 3 steps
  load_shed_correct     (0.15) — shed_load used on Industrial zone, NOT on high/critical zones
"""

from __future__ import annotations

from typing import Any, Dict, List


def grade_medium(final_state: dict, action_history: List[Dict[str, Any]]) -> dict:
    """
    Grade a completed TASK_MEDIUM episode deterministically.

    Args:
        final_state:    env.state()["grid_state"]  (dict form of GridState)
        action_history: env.state()["action_history"]  (list of action dicts)

    Returns:
        {"score": float [0,1], "breakdown": dict, "reason": str}

    Note:
        Criterion 3 (no_line3_overload) requires the full state_history to be
        passed through final_state["_state_history"] if available — the grader
        checks both final_state and any state snapshots found.
        The FastAPI /grade endpoint passes env.state() which includes state_history.
    """
    breakdown: Dict[str, float] = {}
    reasons: List[str] = []

    # Collect all state snapshots for history-sensitive checks
    # The caller may pass env.state() or just final_state; we handle both.
    state_history: List[dict] = final_state.get("_state_history", [])
    # If state_history not embedded, we can only check final state
    all_grid_states: List[dict] = state_history if state_history else [final_state]

    # ------------------------------------------------------------------ #
    # 1. city_south_restored (0.30)
    #    zone-city-south must NOT be blacked out in final grid_state
    # ------------------------------------------------------------------ #
    zones = final_state.get("load_zones", [])
    city_south = next(
        (z for z in zones if z.get("id") == "zone-city-south"), None
    )

    if city_south is None:
        # Zone not found — neutral (task definition issue)
        breakdown["city_south_restored"] = 0.15
        reasons.append("City South zone not found in final state (neutral 0.15).")
    elif not city_south.get("is_blacked_out", True):
        breakdown["city_south_restored"] = 0.30
        served = city_south.get("served_mw", 0)
        demand = city_south.get("demand_mw", 1)
        reasons.append(
            f"City South restored: {served:.0f}/{demand:.0f} MW served, not blacked out."
        )
    else:
        breakdown["city_south_restored"] = 0.0
        reasons.append("City South still blacked out in final state (0/0.30).")

    # ------------------------------------------------------------------ #
    # 2. line2_repair_initiated (0.20)
    #    Action history must contain send_field_crew with line_id="line-2"
    # ------------------------------------------------------------------ #
    crew_sent_to_line2 = any(
        act.get("action_type") == "send_field_crew"
        and act.get("line_id") == "line-2"
        for act in action_history
    )

    if crew_sent_to_line2:
        breakdown["line2_repair_initiated"] = 0.20
        reasons.append("Line 2 repair initiated: field crew dispatched to line-2.")
    else:
        breakdown["line2_repair_initiated"] = 0.0
        reasons.append("No field crew sent to line-2 (0/0.20).")

    # ------------------------------------------------------------------ #
    # 3. no_line3_overload (0.20)
    #    Line3 (line-3) must never have exceeded 100% rated capacity.
    #    Checks all state snapshots stored in state_history.
    # ------------------------------------------------------------------ #
    line3_overloaded_at_step: List[int] = []

    for snap in all_grid_states:
        grid = snap if "lines" in snap else snap.get("grid_state", snap)
        lines = grid.get("lines", [])
        for ln in lines:
            if ln.get("id") == "line-3":
                current_mw = ln.get("current_mw", 0.0) or 0.0
                max_mw = ln.get("max_mw", 1.0) or 1.0
                utilisation = current_mw / max_mw if max_mw > 0 else 0.0
                status = ln.get("status", "healthy")
                if utilisation > 1.0 or status == "overloaded":
                    step_num = grid.get("current_step", -1)
                    line3_overloaded_at_step.append(step_num)

    if not line3_overloaded_at_step:
        breakdown["no_line3_overload"] = 0.20
        reasons.append("Line 3 never exceeded rated capacity — cascade avoided.")
    else:
        breakdown["no_line3_overload"] = 0.0
        reasons.append(
            f"Line 3 exceeded capacity at step(s): {line3_overloaded_at_step} (0/0.20)."
        )

    # ------------------------------------------------------------------ #
    # 4. alert_handling (0.15)
    #    Critical alert (alert-med-1) must be acknowledged within first 3 steps
    # ------------------------------------------------------------------ #
    # Find acknowledge_alert for the critical alert within steps 1–3
    critical_alert_id = "alert-med-1"
    acked_within_3 = False

    for act in action_history:
        step_num = act.get("_step", 999)
        if (
            act.get("action_type") == "acknowledge_alert"
            and act.get("alert_id") == critical_alert_id
            and step_num <= 3
        ):
            acked_within_3 = True
            break

    # Also accept: any critical alert acknowledged within first 3 steps
    if not acked_within_3:
        alerts_final = final_state.get("alerts", [])
        for act in action_history:
            step_num = act.get("_step", 999)
            if act.get("action_type") == "acknowledge_alert" and step_num <= 3:
                alert_id = act.get("alert_id", "")
                # Verify it was a critical alert
                for a in alerts_final:
                    if a.get("id") == alert_id and a.get("severity") == "critical":
                        acked_within_3 = True
                        break

    if acked_within_3:
        breakdown["alert_handling"] = 0.15
        reasons.append("Critical alert acknowledged within first 3 steps.")
    else:
        # Partial: critical alert eventually acknowledged
        alerts_final = final_state.get("alerts", [])
        critical_acked = any(
            a.get("severity") == "critical" and a.get("acknowledged", False)
            for a in alerts_final
        )
        if critical_acked:
            breakdown["alert_handling"] = 0.07
            reasons.append(
                "Critical alert acknowledged (after step 3) — partial credit 0.07/0.15."
            )
        else:
            breakdown["alert_handling"] = 0.0
            reasons.append("Critical alert never acknowledged (0/0.15).")

    # ------------------------------------------------------------------ #
    # 5. load_shed_correct (0.15)
    #    shed_load must be used on zone-industrial (medium priority).
    #    Penalise if shed_load was used on high or critical priority zones.
    # ------------------------------------------------------------------ #
    # Build a map of zone_id → priority from the final state
    zone_priority_map: Dict[str, str] = {
        z.get("id", ""): z.get("priority", "unknown")
        for z in final_state.get("load_zones", [])
    }

    shed_actions = [
        act for act in action_history
        if act.get("action_type") == "shed_load"
    ]

    if not shed_actions:
        breakdown["load_shed_correct"] = 0.0
        reasons.append("No shed_load actions used (0/0.15).")
    else:
        shed_on_industrial = any(
            act.get("zone_id") == "zone-industrial" for act in shed_actions
        )
        shed_on_critical_or_high = any(
            zone_priority_map.get(act.get("zone_id", ""), "low") in ("critical", "high")
            for act in shed_actions
        )

        if shed_on_industrial and not shed_on_critical_or_high:
            breakdown["load_shed_correct"] = 0.15
            reasons.append(
                "Load shedding correctly applied to Industrial zone only."
            )
        elif shed_on_industrial and shed_on_critical_or_high:
            breakdown["load_shed_correct"] = 0.07
            reasons.append(
                "Shed load on Industrial (correct) but also shed high/critical zones "
                "(partial 0.07/0.15)."
            )
        elif shed_on_critical_or_high:
            breakdown["load_shed_correct"] = 0.0
            reasons.append(
                "Shed load applied to high/critical priority zones — 0/0.15."
            )
        else:
            breakdown["load_shed_correct"] = 0.05
            reasons.append(
                "Shed load used but not on Industrial zone — partial 0.05/0.15."
            )

    # ------------------------------------------------------------------ #
    # Composite score
    # ------------------------------------------------------------------ #
    raw_score = sum(breakdown.values())
    final_score = max(0.0, min(1.0, raw_score))

    return {
        "score": round(final_score, 4),
        "breakdown": breakdown,
        "reason": " | ".join(reasons),
    }
