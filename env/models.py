"""
env/models.py — Pydantic v2 data models for grid-ops-env.

All domain types for the Power Grid Emergency Operations Center OpenEnv.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FuelType(str, Enum):
    nuclear = "nuclear"
    gas = "gas"
    hydro = "hydro"
    wind = "wind"
    solar = "solar"


class GeneratorStatus(str, Enum):
    online = "online"
    offline = "offline"
    tripped = "tripped"


class LineStatus(str, Enum):
    healthy = "healthy"
    overloaded = "overloaded"
    faulted = "faulted"


class ZonePriority(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class AlertSeverity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class ActionType(str, Enum):
    dispatch_generator = "dispatch_generator"
    shed_load = "shed_load"
    reroute_line = "reroute_line"
    request_emergency_reserve = "request_emergency_reserve"
    send_field_crew = "send_field_crew"
    acknowledge_alert = "acknowledge_alert"
    issue_public_notice = "issue_public_notice"
    no_op = "no_op"


# ---------------------------------------------------------------------------
# Grid Component Models
# ---------------------------------------------------------------------------


class Generator(BaseModel):
    """Represents a power generation unit."""

    model_config = {"frozen": False}

    id: str = Field(..., description="Unique identifier for the generator")
    name: str = Field(..., description="Human-readable name")
    status: GeneratorStatus = Field(..., description="Operational status")
    current_mw: float = Field(..., ge=0.0, description="Current power output in MW")
    max_mw: float = Field(..., gt=0.0, description="Maximum rated power output in MW")
    min_mw: float = Field(0.0, ge=0.0, description="Minimum stable generation in MW (0 = can be shut down)")
    cost_per_mw: float = Field(..., ge=0.0, description="Operating cost per MW per step")
    fuel_type: FuelType = Field(..., description="Fuel or energy source type")

    @field_validator("current_mw")
    @classmethod
    def current_within_bounds(cls, v: float) -> float:
        # We allow 0 for offline generators; upper bound checked at model_validator
        if v < 0:
            raise ValueError("current_mw must be >= 0")
        return v

    @model_validator(mode="after")
    def current_not_exceed_max(self) -> "Generator":
        if self.current_mw > self.max_mw:
            raise ValueError(
                f"current_mw ({self.current_mw}) cannot exceed max_mw ({self.max_mw})"
            )
        if self.min_mw > self.max_mw:
            raise ValueError(
                f"min_mw ({self.min_mw}) cannot exceed max_mw ({self.max_mw})"
            )
        return self


class TransmissionLine(BaseModel):
    """Represents a high-voltage power transmission corridor."""

    model_config = {"frozen": False}

    id: str = Field(..., description="Unique identifier for the line")
    name: str = Field(..., description="Human-readable name")
    from_zone: str = Field(..., description="Source load zone id")
    to_zone: str = Field(..., description="Destination load zone id")
    current_mw: float = Field(..., ge=0.0, description="Power currently flowing in MW")
    max_mw: float = Field(..., gt=0.0, description="Thermal rating / capacity in MW")
    status: LineStatus = Field(..., description="Line health status")
    repair_steps_remaining: int = Field(
        0, ge=0, description="Number of steps until repair complete (0 = not under repair or already repaired)"
    )

    @model_validator(mode="after")
    def validate_flow_vs_capacity(self) -> "TransmissionLine":
        # Allow overloaded lines to have flow > max (that is the definition of overloaded)
        if self.status == LineStatus.faulted and self.current_mw != 0.0:
            # Faulted lines carry no power
            object.__setattr__(self, "current_mw", 0.0)
        return self


class LoadZone(BaseModel):
    """Represents a geographic demand zone."""

    model_config = {"frozen": False}

    id: str = Field(..., description="Unique identifier for the zone")
    name: str = Field(..., description="Human-readable name")
    demand_mw: float = Field(..., ge=0.0, description="Current power demand in MW")
    served_mw: float = Field(..., ge=0.0, description="Power actually being delivered in MW")
    priority: ZonePriority = Field(..., description="Criticality classification")
    is_blacked_out: bool = Field(False, description="Whether the zone has no power")

    @model_validator(mode="after")
    def served_not_exceed_demand(self) -> "LoadZone":
        if self.served_mw > self.demand_mw:
            raise ValueError(
                f"served_mw ({self.served_mw}) cannot exceed demand_mw ({self.demand_mw})"
            )
        return self


class Alert(BaseModel):
    """Represents an operational alert raised during a simulation."""

    model_config = {"frozen": False}

    id: str = Field(..., description="Unique alert identifier")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    message: str = Field(..., description="Human-readable alert description")
    acknowledged: bool = Field(False, description="Whether the alert has been acknowledged by the agent")
    step_raised: int = Field(..., ge=0, description="Simulation step when the alert was raised")


class WeatherState(BaseModel):
    """Represents current meteorological conditions."""

    model_config = {"frozen": False}

    wind_speed_kph: float = Field(..., ge=0.0, description="Wind speed in km/h")
    temperature_c: float = Field(..., description="Ambient temperature in Celsius")
    storm_active: bool = Field(False, description="Whether a severe weather event is active")
    wind_forecast_mw: float = Field(..., ge=0.0, description="Forecast wind generation in MW")
    solar_forecast_mw: float = Field(..., ge=0.0, description="Forecast solar generation in MW")


# ---------------------------------------------------------------------------
# Composite State Model
# ---------------------------------------------------------------------------


class GridState(BaseModel):
    """Complete snapshot of the power grid at a single point in time."""

    model_config = {"frozen": False}

    generators: List[Generator] = Field(default_factory=list)
    lines: List[TransmissionLine] = Field(default_factory=list)
    load_zones: List[LoadZone] = Field(default_factory=list)
    alerts: List[Alert] = Field(default_factory=list)
    weather: WeatherState
    budget_remaining: float = Field(..., ge=0.0, description="Remaining operational budget ($)")
    current_step: int = Field(0, ge=0, description="Current simulation step number")
    current_hour: int = Field(
        12, ge=0, le=23, description="Current hour of day (0-23, 24-hour clock)"
    )
    total_mw_demanded: float = Field(0.0, ge=0.0, description="Aggregate demand across all zones")
    total_mw_served: float = Field(0.0, ge=0.0, description="Aggregate power delivered across all zones")

    @model_validator(mode="after")
    def compute_totals(self) -> "GridState":
        """Recompute aggregate totals from zone-level data."""
        self.total_mw_demanded = sum(z.demand_mw for z in self.load_zones)
        self.total_mw_served = sum(z.served_mw for z in self.load_zones)
        return self

    @model_validator(mode="after")
    def served_not_exceed_demanded(self) -> "GridState":
        if self.total_mw_served > self.total_mw_demanded + 1e-6:
            raise ValueError(
                f"total_mw_served ({self.total_mw_served:.2f}) cannot exceed "
                f"total_mw_demanded ({self.total_mw_demanded:.2f})"
            )
        return self


# ---------------------------------------------------------------------------
# RL Interface Models
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """Complete observation returned to the agent after each step."""

    model_config = {"frozen": False}

    task_id: str = Field(..., description="Identifier of the active task scenario")
    step: int = Field(..., ge=0, description="Current step number within the episode")
    goal: str = Field(
        default="",
        description="Natural-language description of the task objective",
    )
    grid_state: GridState = Field(..., description="Full grid state snapshot")
    messages: List[str] = Field(default_factory=list, description="Narrative event messages from the last step")
    done: bool = Field(False, description="Whether the episode has terminated")



class Action(BaseModel):
    """
    Agent action model.

    The ``action_type`` field selects which operation to perform.
    Depending on the action type, only certain optional fields are relevant:

    - dispatch_generator  → generator_id, target_mw
    - shed_load           → zone_id, shed_mw
    - reroute_line        → line_id, target_zone_id
    - request_emergency_reserve → reserve_mw, cost_override
    - send_field_crew     → line_id, crew_id
    - acknowledge_alert   → alert_id
    - issue_public_notice → notice_text, affected_zone_ids
    - no_op               → (no additional fields required)
    """

    model_config = {"frozen": False}

    action_type: ActionType = Field(..., description="The type of grid operation to perform")

    # dispatch_generator
    generator_id: Optional[str] = Field(None, description="Target generator ID (dispatch_generator)")
    target_mw: Optional[float] = Field(None, ge=0.0, description="Desired output level in MW (dispatch_generator)")

    # shed_load
    zone_id: Optional[str] = Field(None, description="Target load zone ID (shed_load)")
    shed_mw: Optional[float] = Field(None, ge=0.0, description="Megawatts to curtail from demand (shed_load)")

    # reroute_line
    line_id: Optional[str] = Field(None, description="Transmission line ID to reroute or repair (reroute_line / send_field_crew)")
    target_zone_id: Optional[str] = Field(None, description="Destination zone for rerouted power (reroute_line)")

    # request_emergency_reserve
    reserve_mw: Optional[float] = Field(None, ge=0.0, description="Emergency reserve capacity requested in MW")
    cost_override: Optional[float] = Field(None, ge=0.0, description="Custom cost for emergency reserve procurement")

    # send_field_crew
    crew_id: Optional[str] = Field(None, description="Identifier for the field crew being dispatched")

    # acknowledge_alert
    alert_id: Optional[str] = Field(None, description="Alert ID to acknowledge (acknowledge_alert)")

    # issue_public_notice
    notice_text: Optional[str] = Field(None, description="Content of the public notice message")
    affected_zone_ids: Optional[List[str]] = Field(None, description="Zones referenced in the public notice")

    @model_validator(mode="after")
    def validate_required_fields_for_type(self) -> "Action":
        """Ensure action-type-specific fields are present."""
        at = self.action_type
        if at == ActionType.dispatch_generator:
            if self.generator_id is None:
                raise ValueError("dispatch_generator requires generator_id")
            if self.target_mw is None:
                raise ValueError("dispatch_generator requires target_mw")
        elif at == ActionType.shed_load:
            if self.zone_id is None:
                raise ValueError("shed_load requires zone_id")
            if self.shed_mw is None:
                raise ValueError("shed_load requires shed_mw")
        elif at == ActionType.reroute_line:
            if self.line_id is None:
                raise ValueError("reroute_line requires line_id")
            if self.target_zone_id is None:
                raise ValueError("reroute_line requires target_zone_id")
        elif at == ActionType.request_emergency_reserve:
            if self.reserve_mw is None:
                raise ValueError("request_emergency_reserve requires reserve_mw")
        elif at == ActionType.send_field_crew:
            if self.line_id is None:
                raise ValueError("send_field_crew requires line_id")
        elif at == ActionType.acknowledge_alert:
            if self.alert_id is None:
                raise ValueError("acknowledge_alert requires alert_id")
        elif at == ActionType.issue_public_notice:
            if self.notice_text is None:
                raise ValueError("issue_public_notice requires notice_text")
        return self


class Reward(BaseModel):
    """Reward signal returned after each environment step."""

    model_config = {"frozen": False}

    score: float = Field(..., description="Composite reward score in [0.0, 1.0]")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Weighted contribution of each reward component",
    )
    reason: str = Field(..., description="Human-readable explanation of the reward signal")

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class StepResult(BaseModel):
    """Complete result returned by the environment after executing one action."""

    model_config = {"frozen": False}

    observation: Observation = Field(..., description="Updated environment observation")
    reward: Reward = Field(..., description="Reward signal for this step")
    done: bool = Field(False, description="Whether the episode has ended")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary diagnostic information (not used for training)",
    )
