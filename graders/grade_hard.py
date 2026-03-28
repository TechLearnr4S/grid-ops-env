"""
graders/grade_hard.py — Deterministic grader for TASK_HARD "Cascading Failure".

Criteria and weights:
  hospital_protected   (0.30) — Hospital zone (zone-hospital) NEVER blacked out at ANY step
  cascade_prevented    (0.20) — Line1 and Line2 never reached 'faulted' status in any snapshot
  budget_managed       (0.20) — budget_remaining >= 0 in final state (not bankrupted)
  nuclear_compensated  (0.15) — total generation within 10% of demand by step 8
  communication        (0.15) — issue_public_notice called at least once
                                 AND all critical alerts acknowledged
"""

from __future__ import annotations

from typing import Any, Dict, List


def grade_hard(final_state: dict, action_history: List[Dict[str, Any]]) -> dict:
    """
    Grade a completed TASK_HARD episode deterministically.

    Args:
        final_state:    env.state()["grid_state"] (dict form of GridState).
                        May also be env.state() directly (contains state_history).
        action_history: env.state()["action_history"]

    Returns:
        {"score": float [0,1], "breakdown": dict, "reason": str}

    IMPORTANT:
        Hospitals and cascade checks require ALL STEPS, not just the final state.
        The grader reads state_history from final_state["_state_history"] or
        falls back to env.state()["state_history"] if passed the full env snapshot.
    """
    breakdown: Dict[str, float] = {}
    reasons: List[str] = []

    # Resolve state_history — supports both calling conventions:
    #   grade_hard(env.state()["grid_state"], history)  — no state_history embedded
    #   grade_hard(env.state(), history)                — full env snapshot
    state_history: List[dict] = (
        final_state.get("_state_history")          # injected by /grade endpoint
        or final_state.get("state_history", [])    # if full env.state() passed
    )
    # Resolve the grid_state dict regardless of how caller passed it
    if "load_zones" in final_state:
        grid_final = final_state
    else:
        grid_final = final_state.get("grid_state", final_state)

    # Build the list of all grid state snapshots to inspect
    all_snapshots: List[dict] = []
    for snap in state_history:
        if "load_zones" in snap:
            all_snapshots.append(snap)
        elif "grid_state" in snap:
            all_snapshots.append(snap["grid_state"])

    # If no history available, fall back to final state only
    if not all_snapshots:
        all_snapshots = [grid_final]

    # ------------------------------------------------------------------ #
    # 1. hospital_protected (0.30)
    #    zone-hospital must NEVER be blacked out in ANY state snapshot.
    # ------------------------------------------------------------------ #
    hospital_blackout_steps: List[int] = []

    for snap in all_snapshots:
        zones = snap.get("load_zones", [])
        for z in zones:
            if z.get("id") == "zone-hospital":
                if z.get("is_blacked_out", False):
                    hospital_blackout_steps.append(snap.get("current_step", -1))

    if not hospital_blackout_steps:
        breakdown["hospital_protected"] = 0.30
        reasons.append("Hospital District never lost power — full protection score.")
    else:
        # Partial credit if blackout was brief (1 step only)
        if len(hospital_blackout_steps) == 1:
            breakdown["hospital_protected"] = 0.10
            reasons.append(
                f"Hospital blacked out briefly at step {hospital_blackout_steps} "
                f"(partial 0.10/0.30)."
            )
        else:
            breakdown["hospital_protected"] = 0.0
            reasons.append(
                f"Hospital blacked out at {len(hospital_blackout_steps)} step(s): "
                f"{hospital_blackout_steps} (0/0.30)."
            )

    # ------------------------------------------------------------------ #
    # 2. cascade_prevented (0.20)
    #    Line1 (line-1) and Line2 (line-2) must never reach 'faulted' status.
    # ------------------------------------------------------------------ #
    cascade_lines = {"line-1", "line-2"}
    faulted_events: Dict[str, List[int]] = {lid: [] for lid in cascade_lines}

    for snap in all_snapshots:
        lines = snap.get("lines", [])
        step_num = snap.get("current_step", -1)
        for ln in lines:
            lid = ln.get("id", "")
            if lid in cascade_lines and ln.get("status") == "faulted":
                faulted_events[lid].append(step_num)

    any_faulted = any(len(steps) > 0 for steps in faulted_events.values())

    if not any_faulted:
        breakdown["cascade_prevented"] = 0.20
        reasons.append("Lines 1 and 2 never reached faulted status — cascade prevented.")
    else:
        faulted_summary = {
            lid: steps for lid, steps in faulted_events.items() if steps
        }
        # Partial: only one line faulted
        n_faulted = sum(1 for steps in faulted_events.values() if steps)
        if n_faulted == 1:
            breakdown["cascade_prevented"] = 0.08
            reasons.append(
                f"One monitored line faulted {faulted_summary} "
                f"(partial 0.08/0.20)."
            )
        else:
            breakdown["cascade_prevented"] = 0.0
            reasons.append(
                f"Both Lines 1 and 2 faulted: {faulted_summary} (0/0.20)."
            )

    # ------------------------------------------------------------------ #
    # 3. budget_managed (0.20)
    #    budget_remaining >= 0 in the final grid state (not bankrupted).
    # ------------------------------------------------------------------ #
    budget_remaining = grid_final.get("budget_remaining", 0.0)

    if budget_remaining >= 0.0:
        budget_score = min(0.20, 0.10 + (budget_remaining / 3000.0) * 0.10)
        breakdown["budget_managed"] = round(budget_score, 4)
        reasons.append(
            f"Budget managed: ${budget_remaining:.0f} remaining "
            f"(score {budget_score:.3f}/0.20)."
        )
    else:
        breakdown["budget_managed"] = 0.0
        reasons.append(f"Budget depleted: ${budget_remaining:.0f} (0/0.20).")

    # ------------------------------------------------------------------ #
    # 4. nuclear_compensated (0.15)
    #    By step 8, total online generation must be within 10% of total demand.
    #    We check the earliest state snapshot at or after step 8.
    # ------------------------------------------------------------------ #
    snap_at_step8: dict = {}
    # Find the snapshot closest to step 8 (≥ 8 preferred, else latest available)
    for snap in all_snapshots:
        step_num = snap.get("current_step", -1)
        if step_num >= 8:
            snap_at_step8 = snap
            break

    if not snap_at_step8:
        # Episode ended before step 8 — use final state and check if compensated early
        snap_at_step8 = grid_final

    # Calculate total online generation from generators in the chosen snapshot
    generators = snap_at_step8.get("generators", [])
    total_gen_mw = sum(
        g.get("current_mw", 0.0) or 0.0
        for g in generators
        if g.get("status") == "online"
    )
    demand_at_step8 = snap_at_step8.get("total_mw_demanded", 0.0) or 0.0

    if demand_at_step8 > 0:
        gen_ratio = total_gen_mw / demand_at_step8
        within_10_pct = abs(1.0 - gen_ratio) <= 0.10
    else:
        within_10_pct = True  # no demand → trivially satisfied
        gen_ratio = 1.0

    if within_10_pct:
        breakdown["nuclear_compensated"] = 0.15
        reasons.append(
            f"Nuclear compensated by step 8: generation {total_gen_mw:.0f} MW "
            f"vs demand {demand_at_step8:.0f} MW ({gen_ratio * 100:.1f}% — within ±10%)."
        )
    else:
        # Partial: within 20%
        within_20_pct = abs(1.0 - gen_ratio) <= 0.20
        if within_20_pct:
            breakdown["nuclear_compensated"] = 0.07
            reasons.append(
                f"Nuclear partially compensated: generation {total_gen_mw:.0f} MW "
                f"vs demand {demand_at_step8:.0f} MW ({gen_ratio * 100:.1f}% — within ±20%, "
                f"partial 0.07/0.15)."
            )
        else:
            breakdown["nuclear_compensated"] = 0.0
            reasons.append(
                f"Nuclear not compensated by step 8: only {gen_ratio * 100:.1f}% "
                f"of demand covered (0/0.15)."
            )

    # ------------------------------------------------------------------ #
    # 5. communication (0.15)
    #    issue_public_notice called at least once
    #    AND all critical alerts acknowledged in final state
    # ------------------------------------------------------------------ #
    notice_issued = any(
        act.get("action_type") == "issue_public_notice"
        for act in action_history
    )

    alerts_final = grid_final.get("alerts", [])
    critical_alerts = [a for a in alerts_final if a.get("severity") == "critical"]
    all_critical_acked = all(
        a.get("acknowledged", False) for a in critical_alerts
    ) if critical_alerts else True

    if notice_issued and all_critical_acked:
        breakdown["communication"] = 0.15
        reasons.append(
            "Communication score: public notice issued + all critical alerts acknowledged."
        )
    elif notice_issued or all_critical_acked:
        which = []
        if notice_issued:
            which.append("public notice issued")
        if all_critical_acked:
            which.append("all critical alerts acknowledged")
        breakdown["communication"] = 0.07
        reasons.append(
            f"Partial communication ({', '.join(which)}) — 0.07/0.15."
        )
    else:
        breakdown["communication"] = 0.0
        reasons.append(
            "No public notice issued and critical alerts unacknowledged (0/0.15)."
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
