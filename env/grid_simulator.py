"""
env/grid_simulator.py — Pure-Python grid physics engine for grid-ops-env.

Handles all 8 action types, time-stepping, cascading fault logic,
and reward computation.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

from .models import (
    Action,
    ActionType,
    Alert,
    AlertSeverity,
    Generator,
    GeneratorStatus,
    GridState,
    LineStatus,
    LoadZone,
    Observation,
    Reward,
    TransmissionLine,
    ZonePriority,
)


# ---------------------------------------------------------------------------
# Priority penalty weights — used in reward calculation
# ---------------------------------------------------------------------------

_PRIORITY_WEIGHT: Dict[str, float] = {
    ZonePriority.critical: 1.0,
    ZonePriority.high: 0.6,
    ZonePriority.medium: 0.3,
    ZonePriority.low: 0.1,
}

# Cost charged per MW of emergency reserve procured (default)
_DEFAULT_RESERVE_COST_PER_MW: float = 50.0
# Cost for dispatching a field crew
_CREW_DISPATCH_COST: float = 200.0
# Repair steps required when a crew is sent (steps to complete repair)
_REPAIR_STEPS: int = 3
# Overload cascade threshold — if a line is overloaded for >1 step it trips
_OVERLOAD_CASCADE_STEPS: int = 1  # overloaded ≥ this many steps → trips to faulted


class GridSimulator:
    """
    Stateless grid physics engine.

    All methods accept an immutable GridState and return new state objects;
    the caller is responsible for managing state across steps.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_action(
        self,
        state: GridState,
        action: Action,
    ) -> Tuple[GridState, Dict[str, float], List[str]]:
        """
        Apply a single agent action to the grid state.

        Returns:
            new_state      — updated GridState after action
            reward_parts   — partial reward breakdown dict (filled by caller)
            messages       — narrative event strings describing what happened
        """
        state = self._deep_copy_state(state)
        messages: List[str] = []
        reward_parts: Dict[str, float] = {}

        at = action.action_type

        if at == ActionType.dispatch_generator:
            state, messages = self._dispatch_generator(state, action, messages)

        elif at == ActionType.shed_load:
            state, messages = self._shed_load(state, action, messages)

        elif at == ActionType.reroute_line:
            state, messages = self._reroute_line(state, action, messages)

        elif at == ActionType.request_emergency_reserve:
            state, messages = self._request_emergency_reserve(state, action, messages)

        elif at == ActionType.send_field_crew:
            state, messages = self._send_field_crew(state, action, messages)

        elif at == ActionType.acknowledge_alert:
            state, messages = self._acknowledge_alert(state, action, messages)

        elif at == ActionType.issue_public_notice:
            state, messages = self._issue_public_notice(state, action, messages)

        elif at == ActionType.no_op:
            messages.append("No operation performed this step.")

        # Recompute grid totals after action
        state = self._recompute_totals(state)
        return state, reward_parts, messages

    def tick(
        self,
        state: GridState,
    ) -> Tuple[GridState, List[str]]:
        """
        Advance the simulation by one time step.

        Performs (in order):
          1. Increment step counter and hour
          2. Scale demand by the hour-of-day curve
          3. Progress line repair countdowns
          4. Check overloaded lines → raise alerts or cascade to faulted
          5. Update blackout flags for load zones
          6. Recompute aggregate totals

        Returns:
            new_state — updated GridState
            events    — list of narrative event strings
        """
        state = self._deep_copy_state(state)
        events: List[str] = []

        # 1. Advance counters
        state.current_step += 1
        state.current_hour = (state.current_hour + 1) % 24

        # 2. Scale demand by hour curve
        multiplier = self._demand_curve(state.current_hour)
        for zone in state.load_zones:
            # Scale demand relative to a reference demand stored at step 0.
            # We scale from the current demand value by the ratio of consecutive multipliers.
            prev_hour = (state.current_hour - 1) % 24
            prev_mult = self._demand_curve(prev_hour)
            if prev_mult > 0:
                zone.demand_mw = zone.demand_mw * (multiplier / prev_mult)
            zone.demand_mw = max(0.0, zone.demand_mw)
            # IMPORTANT: cap served_mw at demand_mw after rescaling to prevent
            # model validation errors (served_mw <= demand_mw invariant).
            zone.served_mw = min(zone.served_mw, zone.demand_mw)

        # 3. Progress repairs
        for line in state.lines:
            if line.status == LineStatus.faulted and line.repair_steps_remaining > 0:
                line.repair_steps_remaining -= 1
                if line.repair_steps_remaining == 0:
                    line.status = LineStatus.healthy
                    events.append(
                        f"✅ Line '{line.name}' repair complete — line is back online."
                    )

        # 4. Check overloaded lines
        for line in state.lines:
            if line.status == LineStatus.healthy and line.current_mw > line.max_mw:
                line.status = LineStatus.overloaded
                alert = self._make_alert(
                    state,
                    AlertSeverity.high,
                    f"⚠️ Line '{line.name}' is OVERLOADED ({line.current_mw:.0f} MW / {line.max_mw:.0f} MW rated).",
                )
                state.alerts.append(alert)
                events.append(alert.message)

            elif line.status == LineStatus.overloaded:
                # Cascade: overloaded line trips if still over capacity
                if line.current_mw > line.max_mw:
                    # Save the power flow BEFORE zeroing it — needed to compute impact
                    flow_before_trip = line.current_mw
                    line.status = LineStatus.faulted
                    line.current_mw = 0.0
                    line.repair_steps_remaining = _REPAIR_STEPS
                    alert = self._make_alert(
                        state,
                        AlertSeverity.critical,
                        f"CRITICAL: Line '{line.name}' has TRIPPED due to sustained overload — cascading fault risk!",
                    )
                    state.alerts.append(alert)
                    events.append(alert.message)

                    # Drop power to target zone using the pre-trip flow value
                    target = self._find_zone(state, line.to_zone)
                    if target is not None:
                        lost_mw = min(target.served_mw, flow_before_trip)
                        target.served_mw = max(0.0, target.served_mw - lost_mw)
                        if target.served_mw < target.demand_mw * 0.05:
                            target.is_blacked_out = True
                            events.append(
                                f"BLACKOUT: Load zone '{target.name}' has lost power."
                            )
                else:
                    # Flow is now within limits — recover to healthy
                    line.status = LineStatus.healthy
                    events.append(
                        f"✅ Line '{line.name}' flow normalised — status restored to healthy."
                    )

        # 5. Update blackout flags
        for zone in state.load_zones:
            if zone.served_mw < zone.demand_mw * 0.05:
                if not zone.is_blacked_out:
                    zone.is_blacked_out = True
                    events.append(f"🌑 Zone '{zone.name}' entering blackout.")
            else:
                if zone.is_blacked_out:
                    zone.is_blacked_out = False
                    events.append(f"💡 Zone '{zone.name}' power restored.")

        # 6. Recompute totals
        state = self._recompute_totals(state)
        return state, events

    def calculate_reward(
        self,
        state: GridState,
        prev_state: GridState,
        action: Action,
        messages: List[str],
    ) -> Reward:
        """
        Compute composite reward for a single step transition.

        Component weights:
          0.40 — MW served ratio (how well demand is met)
          0.30 — Blackout penalty weighted by zone priority
          0.20 — Budget efficiency (penalise large budget spend)
          0.10 — Alert response (reward prompt acknowledgement)
        """
        breakdown: Dict[str, float] = {}

        # -- 1. MW served ratio (0.0 – 1.0) --
        if state.total_mw_demanded > 0:
            served_ratio = state.total_mw_served / state.total_mw_demanded
        else:
            served_ratio = 1.0
        served_ratio = max(0.0, min(1.0, served_ratio))
        breakdown["mw_served_ratio"] = round(served_ratio * 0.40, 4)

        # -- 2. Blackout penalty by zone priority (0.0 – 1.0 inverted) --
        total_priority_weight = sum(
            _PRIORITY_WEIGHT.get(z.priority, 0.1) for z in state.load_zones
        )
        blackout_penalty = 0.0
        for zone in state.load_zones:
            if zone.is_blacked_out:
                blackout_penalty += _PRIORITY_WEIGHT.get(zone.priority, 0.1)
        if total_priority_weight > 0:
            blackout_score = 1.0 - (blackout_penalty / total_priority_weight)
        else:
            blackout_score = 1.0
        blackout_score = max(0.0, min(1.0, blackout_score))
        breakdown["blackout_score"] = round(blackout_score * 0.30, 4)

        # -- 3. Budget efficiency (reward frugality) --
        budget_spent = max(0.0, prev_state.budget_remaining - state.budget_remaining)
        if prev_state.budget_remaining > 0:
            budget_use_fraction = budget_spent / prev_state.budget_remaining
        else:
            budget_use_fraction = 1.0
        # Reward staying under budget — penalise heavy spend in a single step
        budget_score = max(0.0, 1.0 - budget_use_fraction * 5.0)
        breakdown["budget_efficiency"] = round(budget_score * 0.20, 4)

        # -- 4. Alert response quality --
        total_alerts = len(state.alerts)
        if total_alerts == 0:
            alert_score = 1.0
        else:
            ack_count = sum(1 for a in state.alerts if a.acknowledged)
            alert_score = ack_count / total_alerts
        breakdown["alert_response"] = round(alert_score * 0.10, 4)

        # -- Composite score --
        raw_score = sum(breakdown.values())
        raw_score = max(0.0, min(1.0, raw_score))

        # Build human-readable reason
        reason_parts = [
            f"MW served: {served_ratio * 100:.1f}% of demand",
            f"blackout score: {blackout_score:.2f}",
            f"budget spent this step: ${budget_spent:.0f}",
            f"alerts acknowledged: {sum(1 for a in state.alerts if a.acknowledged)}/{total_alerts}",
        ]
        reason = " | ".join(reason_parts)

        return Reward(score=raw_score, breakdown=breakdown, reason=reason)

    def check_done(self, state: GridState, max_steps: int) -> bool:
        """
        Return True if the episode should terminate.

        Termination conditions:
          - Reached maximum step count
          - Budget depleted to zero
          - All critical zones are blacked out simultaneously
        """
        if state.current_step >= max_steps:
            return True
        if state.budget_remaining <= 0.0:
            return True
        critical_zones = [z for z in state.load_zones if z.priority == ZonePriority.critical]
        if critical_zones and all(z.is_blacked_out for z in critical_zones):
            return True
        return False

    # ------------------------------------------------------------------
    # Demand curve
    # ------------------------------------------------------------------

    def _demand_curve(self, hour: int) -> float:
        """
        Return a demand multiplier for the given hour of day.

        Profile:
          - Trough: 0.7× at hours 3–5 (overnight low, minimum at hour 4)
          - Peak:   1.4× at hours 18–20 (evening demand surge, maximum at hour 16)
          - Smooth sinusoidal interpolation between peaks and troughs

        Implementation:
          We use a negative cosine so the curve is at its *minimum* when
          (hour - trough_hour) == 0, i.e. the trough falls at hour 4.
          The curvature:
              f(h) = midpoint - amplitude * cos(2π(h - 4) / 24)
          gives minimum 0.70× at h=4 and maximum 1.40× at h=16,
          which sits comfortably within the 15:00–21:00 peak demand window.
        """
        trough_hour = 4          # overnight minimum
        amplitude = (1.4 - 0.7) / 2.0   # 0.35
        midpoint = (1.4 + 0.7) / 2.0    # 1.05
        radians = 2.0 * math.pi * (hour - trough_hour) / 24.0
        # Negative cosine: minimum at trough_hour, maximum 12 h later (hour 16)
        multiplier = midpoint - amplitude * math.cos(radians)
        return round(multiplier, 4)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _dispatch_generator(
        self,
        state: GridState,
        action: Action,
        messages: List[str],
    ) -> Tuple[GridState, List[str]]:
        gen = self._find_generator(state, action.generator_id)
        if gen is None:
            messages.append(f"❌ Generator '{action.generator_id}' not found — action skipped.")
            return state, messages

        target_mw = action.target_mw

        if target_mw == 0.0 and gen.status == GeneratorStatus.online:
            # Shutting down
            cost = gen.current_mw * gen.cost_per_mw * 0.5  # shutdown surcharge
            if state.budget_remaining >= cost:
                state.budget_remaining -= cost
                gen.status = GeneratorStatus.offline
                gen.current_mw = 0.0
                messages.append(
                    f"🔌 Generator '{gen.name}' shut down (cost: ${cost:.0f})."
                )
            else:
                messages.append(
                    f"⚠️ Insufficient budget to shut down '{gen.name}' safely — action skipped."
                )
        elif gen.status in (GeneratorStatus.offline, GeneratorStatus.tripped):
            if target_mw > 0:
                # Startup cost
                startup_cost = gen.min_mw * gen.cost_per_mw * 2.0
                if state.budget_remaining >= startup_cost:
                    state.budget_remaining -= startup_cost
                    gen.status = GeneratorStatus.online
                    gen.current_mw = max(gen.min_mw, min(target_mw, gen.max_mw))
                    messages.append(
                        f"⚡ Generator '{gen.name}' started up at {gen.current_mw:.0f} MW "
                        f"(startup cost: ${startup_cost:.0f})."
                    )
                else:
                    messages.append(
                        f"⚠️ Insufficient budget (${state.budget_remaining:.0f}) to start "
                        f"'{gen.name}' — need ${startup_cost:.0f}."
                    )
        elif gen.status == GeneratorStatus.online:
            # Ramp up or down
            clamped = max(gen.min_mw, min(target_mw, gen.max_mw))
            delta_mw = abs(clamped - gen.current_mw)
            ramp_cost = delta_mw * gen.cost_per_mw
            if state.budget_remaining >= ramp_cost:
                state.budget_remaining -= ramp_cost
                old_mw = gen.current_mw
                gen.current_mw = clamped
                messages.append(
                    f"⚡ Generator '{gen.name}' ramped from {old_mw:.0f} MW → {clamped:.0f} MW "
                    f"(cost: ${ramp_cost:.0f})."
                )
            else:
                messages.append(
                    f"⚠️ Insufficient budget to ramp '{gen.name}' to {target_mw:.0f} MW — action skipped."
                )

        return state, messages

    def _shed_load(
        self,
        state: GridState,
        action: Action,
        messages: List[str],
    ) -> Tuple[GridState, List[str]]:
        zone = self._find_zone(state, action.zone_id)
        if zone is None:
            messages.append(f"❌ Load zone '{action.zone_id}' not found — action skipped.")
            return state, messages

        shed = min(action.shed_mw, zone.served_mw)
        zone.served_mw = max(0.0, zone.served_mw - shed)
        zone.demand_mw = max(zone.served_mw, zone.demand_mw - shed)

        if zone.served_mw < zone.demand_mw * 0.05:
            zone.is_blacked_out = True

        messages.append(
            f"🔻 Load shedding in '{zone.name}': {shed:.0f} MW curtailed "
            f"(now serving {zone.served_mw:.0f}/{zone.demand_mw:.0f} MW)."
        )
        return state, messages

    def _reroute_line(
        self,
        state: GridState,
        action: Action,
        messages: List[str],
    ) -> Tuple[GridState, List[str]]:
        line = self._find_line(state, action.line_id)
        if line is None:
            messages.append(f"❌ Line '{action.line_id}' not found — action skipped.")
            return state, messages

        if line.status == LineStatus.faulted:
            messages.append(
                f"❌ Cannot reroute faulted line '{line.name}' — send a field crew first."
            )
            return state, messages

        target_zone = self._find_zone(state, action.target_zone_id)
        if target_zone is None:
            messages.append(
                f"❌ Target zone '{action.target_zone_id}' not found — reroute skipped."
            )
            return state, messages

        reroute_cost = 100.0
        if state.budget_remaining < reroute_cost:
            messages.append(
                f"⚠️ Insufficient budget for reroute (need ${reroute_cost:.0f}, "
                f"have ${state.budget_remaining:.0f}) — skipped."
            )
            return state, messages

        state.budget_remaining -= reroute_cost
        old_to = line.to_zone
        line.to_zone = action.target_zone_id

        # Transfer power flow to new target zone
        rerouted_mw = min(line.current_mw, target_zone.demand_mw - target_zone.served_mw)
        rerouted_mw = max(0.0, rerouted_mw)

        # Remove supply from old zone
        old_zone = self._find_zone(state, old_to)
        if old_zone is not None:
            old_zone.served_mw = max(0.0, old_zone.served_mw - rerouted_mw)

        target_zone.served_mw = min(
            target_zone.demand_mw, target_zone.served_mw + rerouted_mw
        )
        if target_zone.served_mw >= target_zone.demand_mw * 0.05:
            target_zone.is_blacked_out = False

        messages.append(
            f"🔀 Line '{line.name}' rerouted from zone '{old_to}' → '{action.target_zone_id}': "
            f"{rerouted_mw:.0f} MW redirected (cost: ${reroute_cost:.0f})."
        )
        return state, messages

    def _request_emergency_reserve(
        self,
        state: GridState,
        action: Action,
        messages: List[str],
    ) -> Tuple[GridState, List[str]]:
        cost_per_mw = (
            action.cost_override
            if action.cost_override is not None
            else _DEFAULT_RESERVE_COST_PER_MW
        )
        total_cost = action.reserve_mw * cost_per_mw

        if state.budget_remaining < total_cost:
            affordable_mw = state.budget_remaining / cost_per_mw
            messages.append(
                f"⚠️ Emergency reserve request for {action.reserve_mw:.0f} MW would cost "
                f"${total_cost:.0f} but budget only has ${state.budget_remaining:.0f}. "
                f"Procuring {affordable_mw:.0f} MW instead."
            )
            action = action.model_copy(update={"reserve_mw": affordable_mw})
            total_cost = state.budget_remaining

        state.budget_remaining -= total_cost

        # Add a virtual generator representing the reserve procurement
        remaining = action.reserve_mw
        for zone in state.load_zones:
            needed = zone.demand_mw - zone.served_mw
            if needed > 0 and remaining > 0:
                fill = min(needed, remaining)
                zone.served_mw += fill
                remaining -= fill
                if zone.served_mw >= zone.demand_mw * 0.05:
                    zone.is_blacked_out = False

        messages.append(
            f"🚨 Emergency reserve of {action.reserve_mw:.0f} MW procured at "
            f"${cost_per_mw:.0f}/MW — total cost ${total_cost:.0f}."
        )
        return state, messages

    def _send_field_crew(
        self,
        state: GridState,
        action: Action,
        messages: List[str],
    ) -> Tuple[GridState, List[str]]:
        line = self._find_line(state, action.line_id)
        if line is None:
            messages.append(f"❌ Line '{action.line_id}' not found — cannot deploy crew.")
            return state, messages

        if state.budget_remaining < _CREW_DISPATCH_COST:
            messages.append(
                f"⚠️ Insufficient budget to dispatch crew (need ${_CREW_DISPATCH_COST:.0f}, "
                f"have ${state.budget_remaining:.0f})."
            )
            return state, messages

        state.budget_remaining -= _CREW_DISPATCH_COST

        if line.status != LineStatus.faulted:
            messages.append(
                f"ℹ️ Field crew dispatched to '{line.name}' but line is not faulted "
                f"(status: {line.status.value}). Crew will stand by."
            )
        else:
            if line.repair_steps_remaining == 0:
                line.repair_steps_remaining = _REPAIR_STEPS
            messages.append(
                f"🛠️ Field crew dispatched to '{line.name}' — repair will complete in "
                f"{line.repair_steps_remaining} step(s) (crew cost: ${_CREW_DISPATCH_COST:.0f})."
            )
        return state, messages

    def _acknowledge_alert(
        self,
        state: GridState,
        action: Action,
        messages: List[str],
    ) -> Tuple[GridState, List[str]]:
        alert = self._find_alert(state, action.alert_id)
        if alert is None:
            messages.append(f"❌ Alert '{action.alert_id}' not found — cannot acknowledge.")
            return state, messages

        if alert.acknowledged:
            messages.append(f"ℹ️ Alert '{alert.id}' was already acknowledged.")
        else:
            alert.acknowledged = True
            messages.append(
                f"✅ Alert '{alert.id}' acknowledged: \"{alert.message[:60]}...\""
                if len(alert.message) > 60
                else f"✅ Alert '{alert.id}' acknowledged: \"{alert.message}\""
            )
        return state, messages

    def _issue_public_notice(
        self,
        state: GridState,
        action: Action,
        messages: List[str],
    ) -> Tuple[GridState, List[str]]:
        zones_referenced = action.affected_zone_ids or []
        zone_names = []
        for zid in zones_referenced:
            z = self._find_zone(state, zid)
            zone_names.append(z.name if z else zid)

        notice_cost = 50.0
        if state.budget_remaining >= notice_cost:
            state.budget_remaining -= notice_cost

        zones_str = ", ".join(zone_names) if zone_names else "all zones"
        messages.append(
            f"📢 PUBLIC NOTICE issued for {zones_str}: \"{action.notice_text}\" "
            f"(communication cost: ${notice_cost:.0f})"
        )
        return state, messages

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_copy_state(state: GridState) -> GridState:
        return state.model_copy(deep=True)

    @staticmethod
    def _recompute_totals(state: GridState) -> GridState:
        state.total_mw_demanded = sum(z.demand_mw for z in state.load_zones)
        state.total_mw_served = sum(z.served_mw for z in state.load_zones)
        return state

    @staticmethod
    def _find_generator(state: GridState, gen_id: Optional[str]) -> Optional[Generator]:
        if gen_id is None:
            return None
        for g in state.generators:
            if g.id == gen_id:
                return g
        return None

    @staticmethod
    def _find_line(state: GridState, line_id: Optional[str]) -> Optional[TransmissionLine]:
        if line_id is None:
            return None
        for ln in state.lines:
            if ln.id == line_id:
                return ln
        return None

    @staticmethod
    def _find_zone(state: GridState, zone_id: Optional[str]) -> Optional[LoadZone]:
        if zone_id is None:
            return None
        for z in state.load_zones:
            if z.id == zone_id:
                return z
        return None

    @staticmethod
    def _find_alert(state: GridState, alert_id: Optional[str]) -> Optional[Alert]:
        if alert_id is None:
            return None
        for a in state.alerts:
            if a.id == alert_id:
                return a
        return None

    @staticmethod
    def _make_alert(state: GridState, severity: AlertSeverity, message: str) -> Alert:
        alert_id = f"alert-auto-{state.current_step}-{len(state.alerts)}"
        return Alert(
            id=alert_id,
            severity=severity,
            message=message,
            acknowledged=False,
            step_raised=state.current_step,
        )
