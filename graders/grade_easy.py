"""
graders/grade_easy.py — Deterministic grader for TASK_EASY "Peak Demand Dispatch".

Criteria and weights:
  supply_restored       (0.35) — total_mw_served / total_mw_demanded >= 0.95 in final state
  correct_dispatch      (0.25) — history contains dispatch on gas + hydro both targeted
  alert_acknowledged    (0.15) — all alerts acknowledged in final state
  efficiency            (0.15) — 4 steps → full, 5 → 0.10, 6 → 0.05, >6 → 0.0
  no_unnecessary_actions(0.10) — no duplicate identical actions in history
"""

from __future__ import annotations

from typing import Any, Dict, List


def grade_easy(final_state: dict, action_history: List[Dict[str, Any]]) -> dict:
    """
    Grade a completed TASK_EASY episode deterministically.

    Args:
        final_state:    env.state()["grid_state"]  (dict form of GridState)
        action_history: env.state()["action_history"]  (list of action dicts)

    Returns:
        {"score": float [0,1], "breakdown": dict, "reason": str}
    """
    breakdown: Dict[str, float] = {}
    reasons: List[str] = []

    # ------------------------------------------------------------------ #
    # 1. supply_restored (0.35)
    #    total_mw_served / total_mw_demanded >= 0.95 in final grid state
    # ------------------------------------------------------------------ #
    total_demanded = final_state.get("total_mw_demanded", 0.0)
    total_served = final_state.get("total_mw_served", 0.0)

    if total_demanded > 0:
        served_ratio = total_served / total_demanded
    else:
        served_ratio = 1.0

    if served_ratio >= 0.95:
        breakdown["supply_restored"] = 0.35
        reasons.append(f"Supply restored: {served_ratio * 100:.1f}% of demand met (≥95% target).")
    else:
        # Partial credit: proportional for anything above 70%
        partial = max(0.0, (served_ratio - 0.70) / (0.95 - 0.70)) * 0.35
        breakdown["supply_restored"] = round(partial, 4)
        reasons.append(
            f"Supply partially restored: {served_ratio * 100:.1f}% of demand "
            f"(partial credit {partial:.3f}/0.35)."
        )

    # ------------------------------------------------------------------ #
    # 2. correct_dispatch (0.25)
    #    History must contain dispatch_generator for:
    #      - Riverside Gas Peaker (gen-gas-1)
    #      - Hydro Unit 1 (gen-hydro-1) — must be brought online (target_mw > 0)
    # ------------------------------------------------------------------ #
    dispatched_generators = set()
    hydro_brought_online = False

    for act in action_history:
        if act.get("action_type") == "dispatch_generator":
            gen_id = act.get("generator_id", "")
            target_mw = act.get("target_mw", 0.0) or 0.0
            dispatched_generators.add(gen_id)
            if gen_id == "gen-hydro-1" and target_mw > 0:
                hydro_brought_online = True

    gas_dispatched = "gen-gas-1" in dispatched_generators
    hydro_dispatched = hydro_brought_online

    if gas_dispatched and hydro_dispatched:
        breakdown["correct_dispatch"] = 0.25
        reasons.append("Correct dispatch: gas plant ramped + hydro unit brought online.")
    elif hydro_dispatched or gas_dispatched:
        breakdown["correct_dispatch"] = 0.12
        which = "hydro" if hydro_dispatched else "gas"
        reasons.append(f"Partial dispatch: only {which} plant dispatched (0.12/0.25).")
    else:
        breakdown["correct_dispatch"] = 0.0
        reasons.append("No correct dispatch actions found (0/0.25).")

    # ------------------------------------------------------------------ #
    # 3. alert_acknowledged (0.15)
    #    All alerts in final_state must be acknowledged
    # ------------------------------------------------------------------ #
    alerts = final_state.get("alerts", [])
    if not alerts:
        breakdown["alert_acknowledged"] = 0.15
        reasons.append("No alerts present — full alert score.")
    else:
        total_alerts = len(alerts)
        acked = sum(1 for a in alerts if a.get("acknowledged", False))
        ack_ratio = acked / total_alerts
        score_ack = round(ack_ratio * 0.15, 4)
        breakdown["alert_acknowledged"] = score_ack
        reasons.append(
            f"Alerts acknowledged: {acked}/{total_alerts} "
            f"(score {score_ack:.3f}/0.15)."
        )

    # ------------------------------------------------------------------ #
    # 4. efficiency (0.15)
    #    Full score for ≤4 steps, degrading to 0 for >6
    # ------------------------------------------------------------------ #
    steps_taken = len(action_history)

    if steps_taken <= 4:
        breakdown["efficiency"] = 0.15
        reasons.append(f"Efficiency: solved in {steps_taken} step(s) — full score.")
    elif steps_taken == 5:
        breakdown["efficiency"] = 0.10
        reasons.append("Efficiency: solved in 5 steps — 0.10/0.15.")
    elif steps_taken == 6:
        breakdown["efficiency"] = 0.05
        reasons.append("Efficiency: solved in 6 steps — 0.05/0.15.")
    else:
        breakdown["efficiency"] = 0.0
        reasons.append(f"Efficiency: {steps_taken} steps used — 0/0.15 (over budget).")

    # ------------------------------------------------------------------ #
    # 5. no_unnecessary_actions (0.10)
    #    Penalise repeated identical actions (same action_type + same params)
    # ------------------------------------------------------------------ #
    def _action_fingerprint(act: Dict[str, Any]) -> str:
        """Stable string key for action deduplication (excludes _step metadata)."""
        return "|".join(
            f"{k}={v}"
            for k, v in sorted(act.items())
            if k != "_step" and v is not None
        )

    fingerprints = [_action_fingerprint(a) for a in action_history]
    unique_fps = set(fingerprints)

    if len(fingerprints) == len(unique_fps):
        breakdown["no_unnecessary_actions"] = 0.10
        reasons.append("No repeated actions — full efficiency bonus.")
    else:
        duplicates = len(fingerprints) - len(unique_fps)
        breakdown["no_unnecessary_actions"] = 0.0
        reasons.append(
            f"Found {duplicates} repeated action(s) — 0/0.10 unnecessary action penalty."
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
