"""
tasks/task_definitions.py — Fully defined scenario configurations for grid-ops-env.

Each task returns a ready-to-use GridState object along with task metadata.
All three difficulty levels are provided:
  - TASK_EASY   : Peak Demand Dispatch
  - TASK_MEDIUM : Line Fault Recovery
  - TASK_HARD   : Cascading Failure
"""

from __future__ import annotations

from typing import Any, Dict

from env.models import (
    Alert,
    AlertSeverity,
    FuelType,
    Generator,
    GeneratorStatus,
    GridState,
    LineStatus,
    LoadZone,
    TransmissionLine,
    WeatherState,
    ZonePriority,
)


# ---------------------------------------------------------------------------
# TASK EASY — "Peak Demand Dispatch"
# ---------------------------------------------------------------------------
# Scenario: Evening peak hour. Not enough generation online to meet demand.
# The agent must dispatch the gas plant to full capacity and bring the hydro
# plant online to close the 300 MW supply gap.
# ---------------------------------------------------------------------------

TASK_EASY: Dict[str, Any] = {
    "id": "task_easy",
    "name": "Peak Demand Dispatch",
    "difficulty": "easy",
    "max_steps": 6,
    "description": (
        "It's 18:00 — peak evening demand. Supply is 300 MW short. "
        "Bring Hydro Unit 1 online and ramp the gas plant to full capacity."
    ),
    "initial_state": GridState(
        generators=[
            Generator(
                id="gen-nuclear-1",
                name="Northfield Nuclear",
                status=GeneratorStatus.online,
                current_mw=400.0,
                max_mw=500.0,
                min_mw=300.0,
                cost_per_mw=2.5,
                fuel_type=FuelType.nuclear,
            ),
            Generator(
                id="gen-gas-1",
                name="Riverside Gas Peaker",
                status=GeneratorStatus.online,
                current_mw=200.0,
                max_mw=300.0,
                min_mw=50.0,
                cost_per_mw=8.0,
                fuel_type=FuelType.gas,
            ),
            Generator(
                id="gen-hydro-1",
                name="Hydro Unit 1",
                status=GeneratorStatus.offline,
                current_mw=0.0,
                max_mw=200.0,
                min_mw=20.0,
                cost_per_mw=4.0,
                fuel_type=FuelType.hydro,
            ),
        ],
        lines=[
            TransmissionLine(
                id="line-A",
                name="North–City Main",
                from_zone="zone-supply",
                to_zone="zone-city",
                current_mw=400.0,
                max_mw=600.0,
                status=LineStatus.healthy,
                repair_steps_remaining=0,
            ),
            TransmissionLine(
                id="line-B",
                name="River–Industrial Feeder",
                from_zone="zone-supply",
                to_zone="zone-industrial",
                current_mw=200.0,
                max_mw=400.0,
                status=LineStatus.healthy,
                repair_steps_remaining=0,
            ),
        ],
        load_zones=[
            LoadZone(
                id="zone-city",
                name="City Centre",
                demand_mw=600.0,
                served_mw=400.0,
                priority=ZonePriority.high,
                is_blacked_out=False,
            ),
            LoadZone(
                id="zone-industrial",
                name="Industrial Park",
                demand_mw=300.0,
                served_mw=200.0,
                priority=ZonePriority.medium,
                is_blacked_out=False,
            ),
        ],
        alerts=[
            Alert(
                id="alert-easy-1",
                severity=AlertSeverity.medium,
                message="Demand approaching capacity limits — generation headroom below 10%.",
                acknowledged=False,
                step_raised=0,
            )
        ],
        weather=WeatherState(
            wind_speed_kph=15.0,
            temperature_c=28.0,
            storm_active=False,
            wind_forecast_mw=0.0,
            solar_forecast_mw=0.0,
        ),
        budget_remaining=10_000.0,
        current_step=0,
        current_hour=18,
        total_mw_demanded=0.0,   # recomputed by model_validator
        total_mw_served=0.0,     # recomputed by model_validator
    ),
}


# ---------------------------------------------------------------------------
# TASK MEDIUM — "Line Fault Recovery"
# ---------------------------------------------------------------------------
# Scenario: A transmission line has faulted, leaving City South blacked out.
# Line 3 is a viable bypass but is already at 85% capacity. The agent must
# acknowledge alerts, shed 50 MW from Industrial to free headroom on Line 3,
# reroute power through Line 3 to City South, and dispatch a repair crew for
# the faulted line.
# ---------------------------------------------------------------------------

TASK_MEDIUM: Dict[str, Any] = {
    "id": "task_medium",
    "name": "Line Fault Recovery",
    "difficulty": "medium",
    "max_steps": 10,
    "description": (
        "Line 2 has faulted — City South is blacked out. "
        "Reroute power via Line 3 and dispatch a repair crew. "
        "Be careful: Line 3 is already at 85% capacity."
    ),
    "initial_state": GridState(
        generators=[
            Generator(
                id="gen-nuclear-1",
                name="Northfield Nuclear",
                status=GeneratorStatus.online,
                current_mw=450.0,
                max_mw=500.0,
                min_mw=300.0,
                cost_per_mw=2.5,
                fuel_type=FuelType.nuclear,
            ),
            Generator(
                id="gen-gas-1",
                name="Riverside Gas Peaker",
                status=GeneratorStatus.online,
                current_mw=200.0,
                max_mw=300.0,
                min_mw=50.0,
                cost_per_mw=8.0,
                fuel_type=FuelType.gas,
            ),
            Generator(
                id="gen-hydro-1",
                name="Hydro Unit 1",
                status=GeneratorStatus.online,
                current_mw=150.0,
                max_mw=200.0,
                min_mw=20.0,
                cost_per_mw=4.0,
                fuel_type=FuelType.hydro,
            ),
        ],
        lines=[
            TransmissionLine(
                id="line-1",
                name="North–City North",
                from_zone="zone-supply",
                to_zone="zone-city-north",
                current_mw=300.0,
                max_mw=400.0,
                status=LineStatus.healthy,
                repair_steps_remaining=0,
            ),
            TransmissionLine(
                id="line-2",
                name="River–City South (FAULTED)",
                from_zone="zone-supply",
                to_zone="zone-city-south",
                current_mw=0.0,       # faulted — carries no power
                max_mw=250.0,
                status=LineStatus.faulted,
                repair_steps_remaining=0,  # no crew yet
            ),
            TransmissionLine(
                id="line-3",
                name="Hydro–Industrial Ring",
                from_zone="zone-supply",
                to_zone="zone-industrial",
                current_mw=212.0,     # 85% of 250 MW capacity
                max_mw=250.0,
                status=LineStatus.healthy,
                repair_steps_remaining=0,
            ),
        ],
        load_zones=[
            LoadZone(
                id="zone-city-north",
                name="City North",
                demand_mw=300.0,
                served_mw=300.0,
                priority=ZonePriority.high,
                is_blacked_out=False,
            ),
            LoadZone(
                id="zone-city-south",
                name="City South",
                demand_mw=200.0,
                served_mw=0.0,
                priority=ZonePriority.high,
                is_blacked_out=True,
            ),
            LoadZone(
                id="zone-industrial",
                name="Industrial Park",
                demand_mw=250.0,
                served_mw=250.0,
                priority=ZonePriority.medium,
                is_blacked_out=False,
            ),
        ],
        alerts=[
            Alert(
                id="alert-med-1",
                severity=AlertSeverity.critical,
                message="CRITICAL: Line 2 (River–City South) has FAULTED — City South is without power.",
                acknowledged=False,
                step_raised=0,
            ),
            Alert(
                id="alert-med-2",
                severity=AlertSeverity.high,
                message="WARNING: Line 3 (Hydro–Industrial Ring) operating at 85% capacity — overload risk if rerouted.",
                acknowledged=False,
                step_raised=0,
            ),
        ],
        weather=WeatherState(
            wind_speed_kph=22.0,
            temperature_c=24.0,
            storm_active=False,
            wind_forecast_mw=0.0,
            solar_forecast_mw=30.0,
        ),
        budget_remaining=5_000.0,
        current_step=0,
        current_hour=14,
        total_mw_demanded=0.0,
        total_mw_served=0.0,
    ),
}


# ---------------------------------------------------------------------------
# TASK HARD — "Cascading Failure"
# ---------------------------------------------------------------------------
# Scenario: A nuclear plant has tripped during evening peak, two transmission
# lines are already overloaded, and a severe storm creates wind uncertainty.
# Budget is tight. The agent must prioritise the hospital zone, prevent cascade
# failures on Lines 1 and 2, and carefully manage the dwindling budget.
# ---------------------------------------------------------------------------

TASK_HARD: Dict[str, Any] = {
    "id": "task_hard",
    "name": "Cascading Failure",
    "difficulty": "hard",
    "max_steps": 15,
    "description": (
        "EMERGENCY: Northfield Nuclear has TRIPPED offline during peak evening load. "
        "Lines 1 and 2 are overloaded. A severe storm makes wind output unpredictable. "
        "Protect the hospital at all costs. Budget is critically limited."
    ),
    "initial_state": GridState(
        generators=[
            Generator(
                id="gen-nuclear-1",
                name="Northfield Nuclear",
                status=GeneratorStatus.tripped,
                current_mw=0.0,
                max_mw=500.0,
                min_mw=300.0,
                cost_per_mw=2.5,
                fuel_type=FuelType.nuclear,
            ),
            Generator(
                id="gen-gas-A",
                name="Gas Peaker A",
                status=GeneratorStatus.online,
                current_mw=300.0,
                max_mw=300.0,
                min_mw=50.0,
                cost_per_mw=8.0,
                fuel_type=FuelType.gas,
            ),
            Generator(
                id="gen-gas-B",
                name="Gas Peaker B",
                status=GeneratorStatus.online,
                current_mw=200.0,
                max_mw=200.0,
                min_mw=30.0,
                cost_per_mw=9.5,
                fuel_type=FuelType.gas,
            ),
            Generator(
                id="gen-wind-1",
                name="Coastal Wind Farm",
                status=GeneratorStatus.online,
                current_mw=80.0,      # actual: storm reduces to 80 MW despite 150 MW forecast
                max_mw=150.0,
                min_mw=0.0,
                cost_per_mw=1.0,
                fuel_type=FuelType.wind,
            ),
        ],
        lines=[
            TransmissionLine(
                id="line-1",
                name="Main Artery North",
                from_zone="zone-supply",
                to_zone="zone-city",
                current_mw=440.0,   # 110% of 400 MW rated capacity
                max_mw=400.0,
                status=LineStatus.overloaded,
                repair_steps_remaining=0,
            ),
            TransmissionLine(
                id="line-2",
                name="Industrial Feeder",
                from_zone="zone-supply",
                to_zone="zone-industrial",
                current_mw=315.0,   # 105% of 300 MW rated capacity
                max_mw=300.0,
                status=LineStatus.overloaded,
                repair_steps_remaining=0,
            ),
            TransmissionLine(
                id="line-3",
                name="Hospital Dedicated Feed",
                from_zone="zone-supply",
                to_zone="zone-hospital",
                current_mw=95.0,
                max_mw=150.0,
                status=LineStatus.healthy,
                repair_steps_remaining=0,
            ),
            TransmissionLine(
                id="line-4",
                name="Suburban Distribution",
                from_zone="zone-supply",
                to_zone="zone-suburbs",
                current_mw=140.0,
                max_mw=250.0,
                status=LineStatus.healthy,
                repair_steps_remaining=0,
            ),
        ],
        load_zones=[
            LoadZone(
                id="zone-hospital",
                name="Hospital District",
                demand_mw=100.0,
                served_mw=95.0,
                priority=ZonePriority.critical,
                is_blacked_out=False,
            ),
            LoadZone(
                id="zone-city",
                name="City Centre",
                demand_mw=400.0,
                served_mw=380.0,
                priority=ZonePriority.high,
                is_blacked_out=False,
            ),
            LoadZone(
                id="zone-industrial",
                name="Industrial Zone",
                demand_mw=300.0,
                served_mw=275.0,
                priority=ZonePriority.medium,
                is_blacked_out=False,
            ),
            LoadZone(
                id="zone-suburbs",
                name="Suburban Residential",
                demand_mw=200.0,
                served_mw=140.0,
                priority=ZonePriority.low,
                is_blacked_out=False,
            ),
        ],
        alerts=[
            Alert(
                id="alert-hard-1",
                severity=AlertSeverity.critical,
                message="CRITICAL: Northfield Nuclear (500 MW) has TRIPPED — 500 MW generation lost instantly.",
                acknowledged=False,
                step_raised=0,
            ),
            Alert(
                id="alert-hard-2",
                severity=AlertSeverity.high,
                message="OVERLOAD: Main Artery North carrying 110% of rated capacity — cascade imminent.",
                acknowledged=False,
                step_raised=0,
            ),
            Alert(
                id="alert-hard-3",
                severity=AlertSeverity.high,
                message="OVERLOAD: Industrial Feeder carrying 105% of rated capacity — cascade risk.",
                acknowledged=False,
                step_raised=0,
            ),
            Alert(
                id="alert-hard-4",
                severity=AlertSeverity.medium,
                message="WEATHER: Active storm reducing Coastal Wind Farm output — forecast unreliable.",
                acknowledged=False,
                step_raised=0,
            ),
        ],
        weather=WeatherState(
            wind_speed_kph=85.0,
            temperature_c=18.0,
            storm_active=True,
            wind_forecast_mw=150.0,   # forecast (inaccurate due to storm)
            solar_forecast_mw=0.0,    # night
        ),
        budget_remaining=3_000.0,
        current_step=0,
        current_hour=20,
        total_mw_demanded=0.0,
        total_mw_served=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------


def get_task_by_id(task_id: str) -> Dict[str, Any]:
    """Return the task config dict for the given task_id."""
    _registry = {
        "task_easy": TASK_EASY,
        "task_medium": TASK_MEDIUM,
        "task_hard": TASK_HARD,
    }
    task = _registry.get(task_id)
    if task is None:
        raise KeyError(
            f"Unknown task_id '{task_id}'. Available: {list(_registry.keys())}"
        )
    return task
