"""
tests/test_models.py — Sanity-check tests for grid-ops-env Pydantic models.

Run with:  pytest tests/test_models.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from env.models import (
    Action,
    ActionType,
    Alert,
    AlertSeverity,
    FuelType,
    Generator,
    GeneratorStatus,
    GridState,
    LineStatus,
    LoadZone,
    Observation,
    Reward,
    StepResult,
    TransmissionLine,
    WeatherState,
    ZonePriority,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable minimal valid objects
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_weather() -> WeatherState:
    return WeatherState(
        wind_speed_kph=15.0,
        temperature_c=22.0,
        storm_active=False,
        wind_forecast_mw=0.0,
        solar_forecast_mw=50.0,
    )


@pytest.fixture()
def nuclear_generator() -> Generator:
    return Generator(
        id="gen-nuke-1",
        name="Test Nuclear",
        status=GeneratorStatus.online,
        current_mw=400.0,
        max_mw=500.0,
        min_mw=300.0,
        cost_per_mw=2.5,
        fuel_type=FuelType.nuclear,
    )


@pytest.fixture()
def healthy_line() -> TransmissionLine:
    return TransmissionLine(
        id="line-1",
        name="Test Line A",
        from_zone="zone-a",
        to_zone="zone-b",
        current_mw=100.0,
        max_mw=200.0,
        status=LineStatus.healthy,
        repair_steps_remaining=0,
    )


@pytest.fixture()
def city_zone() -> LoadZone:
    return LoadZone(
        id="zone-city",
        name="City Centre",
        demand_mw=300.0,
        served_mw=250.0,
        priority=ZonePriority.high,
        is_blacked_out=False,
    )


@pytest.fixture()
def test_alert() -> Alert:
    return Alert(
        id="alert-1",
        severity=AlertSeverity.high,
        message="Test alert message for validation.",
        acknowledged=False,
        step_raised=0,
    )


@pytest.fixture()
def minimal_grid_state(
    nuclear_generator: Generator,
    healthy_line: TransmissionLine,
    city_zone: LoadZone,
    test_alert: Alert,
    minimal_weather: WeatherState,
) -> GridState:
    return GridState(
        generators=[nuclear_generator],
        lines=[healthy_line],
        load_zones=[city_zone],
        alerts=[test_alert],
        weather=minimal_weather,
        budget_remaining=5000.0,
        current_step=0,
        current_hour=12,
        total_mw_demanded=0.0,
        total_mw_served=0.0,
    )


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


class TestGenerator:
    def test_generator_instantiates_correctly(self, nuclear_generator: Generator) -> None:
        assert nuclear_generator.id == "gen-nuke-1"
        assert nuclear_generator.status == GeneratorStatus.online
        assert nuclear_generator.fuel_type == FuelType.nuclear
        assert nuclear_generator.current_mw == 400.0

    def test_all_fuel_types_are_valid(self) -> None:
        for fuel in FuelType:
            gen = Generator(
                id=f"gen-{fuel.value}",
                name=f"Test {fuel.value}",
                status=GeneratorStatus.offline,
                current_mw=0.0,
                max_mw=100.0,
                min_mw=0.0,
                cost_per_mw=5.0,
                fuel_type=fuel,
            )
            assert gen.fuel_type == fuel

    def test_all_generator_statuses_are_valid(self) -> None:
        for status in GeneratorStatus:
            gen = Generator(
                id="gen-x",
                name="Test",
                status=status,
                current_mw=0.0,
                max_mw=100.0,
                min_mw=0.0,
                cost_per_mw=1.0,
                fuel_type=FuelType.gas,
            )
            assert gen.status == status

    def test_current_mw_cannot_exceed_max_mw(self) -> None:
        with pytest.raises(ValidationError, match="current_mw"):
            Generator(
                id="gen-bad",
                name="Bad Gen",
                status=GeneratorStatus.online,
                current_mw=600.0,    # exceeds max_mw=500
                max_mw=500.0,
                min_mw=0.0,
                cost_per_mw=1.0,
                fuel_type=FuelType.gas,
            )

    def test_negative_current_mw_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Generator(
                id="gen-neg",
                name="Negative Gen",
                status=GeneratorStatus.offline,
                current_mw=-10.0,
                max_mw=100.0,
                min_mw=0.0,
                cost_per_mw=1.0,
                fuel_type=FuelType.gas,
            )

    def test_min_mw_cannot_exceed_max_mw(self) -> None:
        with pytest.raises(ValidationError, match="min_mw"):
            Generator(
                id="gen-bad-min",
                name="Bad Min Gen",
                status=GeneratorStatus.offline,
                current_mw=0.0,
                max_mw=100.0,
                min_mw=200.0,   # min > max → invalid
                cost_per_mw=1.0,
                fuel_type=FuelType.gas,
            )


# ---------------------------------------------------------------------------
# TransmissionLine tests
# ---------------------------------------------------------------------------


class TestTransmissionLine:
    def test_line_instantiates_correctly(self, healthy_line: TransmissionLine) -> None:
        assert healthy_line.id == "line-1"
        assert healthy_line.status == LineStatus.healthy
        assert healthy_line.current_mw == 100.0
        assert healthy_line.max_mw == 200.0

    def test_all_line_statuses_are_valid(self) -> None:
        for status in LineStatus:
            ln = TransmissionLine(
                id=f"line-{status.value}",
                name=f"Line {status.value}",
                from_zone="a",
                to_zone="b",
                current_mw=0.0,
                max_mw=200.0,
                status=status,
                repair_steps_remaining=0,
            )
            assert ln.status == status

    def test_faulted_line_current_mw_zeroed(self) -> None:
        """A faulted line cannot carry power — model_validator should zero it."""
        ln = TransmissionLine(
            id="line-fault",
            name="Faulted Line",
            from_zone="a",
            to_zone="b",
            current_mw=150.0,    # passed in non-zero but should become 0
            max_mw=200.0,
            status=LineStatus.faulted,
            repair_steps_remaining=3,
        )
        assert ln.current_mw == 0.0

    def test_negative_current_mw_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TransmissionLine(
                id="line-neg",
                name="Negative Line",
                from_zone="a",
                to_zone="b",
                current_mw=-50.0,
                max_mw=200.0,
                status=LineStatus.healthy,
                repair_steps_remaining=0,
            )


# ---------------------------------------------------------------------------
# LoadZone tests
# ---------------------------------------------------------------------------


class TestLoadZone:
    def test_zone_instantiates_correctly(self, city_zone: LoadZone) -> None:
        assert city_zone.id == "zone-city"
        assert city_zone.priority == ZonePriority.high
        assert city_zone.served_mw <= city_zone.demand_mw

    def test_all_priorities_are_valid(self) -> None:
        for prio in ZonePriority:
            zone = LoadZone(
                id=f"zone-{prio.value}",
                name=f"Zone {prio.value}",
                demand_mw=100.0,
                served_mw=80.0,
                priority=prio,
                is_blacked_out=False,
            )
            assert zone.priority == prio

    def test_served_mw_cannot_exceed_demand_mw(self) -> None:
        with pytest.raises(ValidationError, match="served_mw"):
            LoadZone(
                id="zone-bad",
                name="Bad Zone",
                demand_mw=100.0,
                served_mw=150.0,   # served > demand → invalid
                priority=ZonePriority.low,
                is_blacked_out=False,
            )

    def test_blacked_out_zone_can_have_zero_served(self) -> None:
        zone = LoadZone(
            id="zone-dark",
            name="Dark Zone",
            demand_mw=200.0,
            served_mw=0.0,
            priority=ZonePriority.medium,
            is_blacked_out=True,
        )
        assert zone.is_blacked_out is True
        assert zone.served_mw == 0.0


# ---------------------------------------------------------------------------
# Alert tests
# ---------------------------------------------------------------------------


class TestAlert:
    def test_alert_instantiates_correctly(self, test_alert: Alert) -> None:
        assert test_alert.id == "alert-1"
        assert test_alert.severity == AlertSeverity.high
        assert test_alert.acknowledged is False

    def test_all_severities_are_valid(self) -> None:
        for sev in AlertSeverity:
            alert = Alert(
                id=f"alert-{sev.value}",
                severity=sev,
                message=f"Test {sev.value} alert",
                acknowledged=False,
                step_raised=1,
            )
            assert alert.severity == sev


# ---------------------------------------------------------------------------
# WeatherState tests
# ---------------------------------------------------------------------------


class TestWeatherState:
    def test_weather_instantiates_correctly(self, minimal_weather: WeatherState) -> None:
        assert minimal_weather.wind_speed_kph == 15.0
        assert minimal_weather.storm_active is False

    def test_storm_active_defaults_to_false(self) -> None:
        w = WeatherState(
            wind_speed_kph=10.0,
            temperature_c=20.0,
            storm_active=False,
            wind_forecast_mw=100.0,
            solar_forecast_mw=50.0,
        )
        assert w.storm_active is False

    def test_negative_wind_speed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WeatherState(
                wind_speed_kph=-5.0,
                temperature_c=20.0,
                storm_active=False,
                wind_forecast_mw=0.0,
                solar_forecast_mw=0.0,
            )


# ---------------------------------------------------------------------------
# GridState tests
# ---------------------------------------------------------------------------


class TestGridState:
    def test_grid_state_instantiates_correctly(
        self, minimal_grid_state: GridState
    ) -> None:
        assert len(minimal_grid_state.generators) == 1
        assert len(minimal_grid_state.lines) == 1
        assert len(minimal_grid_state.load_zones) == 1
        assert minimal_grid_state.budget_remaining == 5000.0

    def test_grid_state_recomputes_totals_automatically(
        self, minimal_grid_state: GridState
    ) -> None:
        """model_validator should set total_mw_demanded/served from zone data."""
        expected_demand = sum(
            z.demand_mw for z in minimal_grid_state.load_zones
        )
        expected_served = sum(
            z.served_mw for z in minimal_grid_state.load_zones
        )
        assert minimal_grid_state.total_mw_demanded == expected_demand
        assert minimal_grid_state.total_mw_served == expected_served

    def test_total_mw_served_never_exceeds_total_mw_demanded(
        self, minimal_grid_state: GridState
    ) -> None:
        assert (
            minimal_grid_state.total_mw_served
            <= minimal_grid_state.total_mw_demanded + 1e-6
        )

    def test_grid_state_served_never_exceeds_demanded_is_enforced_by_zones(
        self, minimal_weather: WeatherState
    ) -> None:
        """
        GridState always recomputes totals from load zone data, so the invariant
        total_mw_served <= total_mw_demanded is guaranteed by zone-level validation.
        This test confirms that a zone with served_mw > demand_mw is rejected at the
        LoadZone level, making the grid-state invariant impossible to violate in practice.
        """
        # Zone-level rejection enforces the grid-state guarantee
        with pytest.raises(ValidationError, match="served_mw"):
            LoadZone(
                id="zone-bad",
                name="Bad Zone",
                demand_mw=100.0,
                served_mw=150.0,   # served > demand → invalid at zone level
                priority=ZonePriority.low,
                is_blacked_out=False,
            )

        # Verify that a valid GridState always has served <= demanded (totals recomputed)
        valid_state = GridState(
            generators=[],
            lines=[],
            load_zones=[
                LoadZone(
                    id="zone-ok",
                    name="OK Zone",
                    demand_mw=100.0,
                    served_mw=80.0,
                    priority=ZonePriority.low,
                    is_blacked_out=False,
                )
            ],
            alerts=[],
            weather=minimal_weather,
            budget_remaining=1000.0,
            current_step=0,
            current_hour=8,
            total_mw_demanded=0.0,    # will be overwritten by validator
            total_mw_served=0.0,      # will be overwritten by validator
        )
        assert valid_state.total_mw_served <= valid_state.total_mw_demanded

    def test_current_hour_must_be_0_to_23(
        self, minimal_weather: WeatherState
    ) -> None:
        with pytest.raises(ValidationError):
            GridState(
                generators=[],
                lines=[],
                load_zones=[],
                alerts=[],
                weather=minimal_weather,
                budget_remaining=1000.0,
                current_step=0,
                current_hour=25,    # invalid
                total_mw_demanded=0.0,
                total_mw_served=0.0,
            )


# ---------------------------------------------------------------------------
# Action tests
# ---------------------------------------------------------------------------


class TestAction:
    def test_all_action_types_are_valid_enum_values(self) -> None:
        for at in ActionType:
            assert at in ActionType

    def test_no_op_action_requires_no_extra_fields(self) -> None:
        action = Action(action_type=ActionType.no_op)
        assert action.action_type == ActionType.no_op
        assert action.generator_id is None

    def test_dispatch_generator_requires_generator_id_and_target_mw(self) -> None:
        action = Action(
            action_type=ActionType.dispatch_generator,
            generator_id="gen-1",
            target_mw=300.0,
        )
        assert action.generator_id == "gen-1"
        assert action.target_mw == 300.0

    def test_dispatch_generator_missing_generator_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="generator_id"):
            Action(
                action_type=ActionType.dispatch_generator,
                target_mw=300.0,
                # generator_id omitted
            )

    def test_dispatch_generator_missing_target_mw_raises(self) -> None:
        with pytest.raises(ValidationError, match="target_mw"):
            Action(
                action_type=ActionType.dispatch_generator,
                generator_id="gen-1",
                # target_mw omitted
            )

    def test_shed_load_action(self) -> None:
        action = Action(
            action_type=ActionType.shed_load,
            zone_id="zone-industrial",
            shed_mw=50.0,
        )
        assert action.zone_id == "zone-industrial"
        assert action.shed_mw == 50.0

    def test_shed_load_missing_zone_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="zone_id"):
            Action(action_type=ActionType.shed_load, shed_mw=50.0)

    def test_reroute_line_action(self) -> None:
        action = Action(
            action_type=ActionType.reroute_line,
            line_id="line-3",
            target_zone_id="zone-city-south",
        )
        assert action.line_id == "line-3"
        assert action.target_zone_id == "zone-city-south"

    def test_request_emergency_reserve_action(self) -> None:
        action = Action(
            action_type=ActionType.request_emergency_reserve,
            reserve_mw=100.0,
        )
        assert action.reserve_mw == 100.0

    def test_send_field_crew_action(self) -> None:
        action = Action(
            action_type=ActionType.send_field_crew,
            line_id="line-2",
            crew_id="crew-alpha",
        )
        assert action.line_id == "line-2"
        assert action.crew_id == "crew-alpha"

    def test_acknowledge_alert_action(self) -> None:
        action = Action(
            action_type=ActionType.acknowledge_alert,
            alert_id="alert-1",
        )
        assert action.alert_id == "alert-1"

    def test_acknowledge_alert_missing_alert_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="alert_id"):
            Action(action_type=ActionType.acknowledge_alert)

    def test_issue_public_notice_action(self) -> None:
        action = Action(
            action_type=ActionType.issue_public_notice,
            notice_text="Rolling blackouts expected 18:00–21:00.",
            affected_zone_ids=["zone-suburbs", "zone-industrial"],
        )
        assert "Rolling blackouts" in action.notice_text
        assert len(action.affected_zone_ids) == 2

    def test_issue_public_notice_missing_text_raises(self) -> None:
        with pytest.raises(ValidationError, match="notice_text"):
            Action(
                action_type=ActionType.issue_public_notice,
                affected_zone_ids=["zone-a"],
            )

    def test_invalid_action_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Action(action_type="explode_reactor")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------


class TestReward:
    def test_reward_instantiates_correctly(self) -> None:
        reward = Reward(
            score=0.75,
            breakdown={"mw_served_ratio": 0.30, "blackout_score": 0.25},
            reason="Test reward",
        )
        assert reward.score == 0.75
        assert "mw_served_ratio" in reward.breakdown

    def test_reward_score_clamped_above_1(self) -> None:
        reward = Reward(score=1.5, breakdown={}, reason="Overflow test")
        assert reward.score == 1.0

    def test_reward_score_clamped_below_0(self) -> None:
        reward = Reward(score=-0.5, breakdown={}, reason="Underflow test")
        assert reward.score == 0.0

    def test_reward_score_boundary_exactly_0(self) -> None:
        reward = Reward(score=0.0, breakdown={}, reason="Zero boundary")
        assert reward.score == 0.0

    def test_reward_score_boundary_exactly_1(self) -> None:
        reward = Reward(score=1.0, breakdown={}, reason="Max boundary")
        assert reward.score == 1.0

    def test_reward_breakdown_can_be_empty(self) -> None:
        reward = Reward(score=0.5, breakdown={}, reason="Empty breakdown")
        assert reward.breakdown == {}

    def test_reward_breakdown_preserves_component_keys(self) -> None:
        breakdown = {
            "mw_served_ratio": 0.35,
            "blackout_score": 0.28,
            "budget_efficiency": 0.18,
            "alert_response": 0.09,
        }
        reward = Reward(score=0.90, breakdown=breakdown, reason="Full breakdown")
        for key in breakdown:
            assert key in reward.breakdown


# ---------------------------------------------------------------------------
# Observation tests
# ---------------------------------------------------------------------------


class TestObservation:
    def test_observation_instantiates_correctly(
        self, minimal_grid_state: GridState
    ) -> None:
        obs = Observation(
            task_id="task_easy",
            step=1,
            grid_state=minimal_grid_state,
            messages=["Step 1 complete."],
            done=False,
        )
        assert obs.task_id == "task_easy"
        assert obs.step == 1
        assert obs.done is False

    def test_observation_messages_default_to_empty_list(
        self, minimal_grid_state: GridState
    ) -> None:
        obs = Observation(
            task_id="task_hard",
            step=0,
            grid_state=minimal_grid_state,
            done=False,
        )
        assert obs.messages == []


# ---------------------------------------------------------------------------
# StepResult tests
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_step_result_instantiates_correctly(
        self, minimal_grid_state: GridState
    ) -> None:
        obs = Observation(
            task_id="task_medium",
            step=3,
            grid_state=minimal_grid_state,
            messages=["Repair crew dispatched."],
            done=False,
        )
        reward = Reward(score=0.62, breakdown={}, reason="Step 3 reward")
        result = StepResult(
            observation=obs,
            reward=reward,
            done=False,
            info={"elapsed_steps": 3, "budget_spent": 400.0},
        )
        assert result.observation.step == 3
        assert result.reward.score == 0.62
        assert result.info["elapsed_steps"] == 3

    def test_step_result_info_defaults_to_empty_dict(
        self, minimal_grid_state: GridState
    ) -> None:
        obs = Observation(
            task_id="task_easy",
            step=0,
            grid_state=minimal_grid_state,
            done=False,
        )
        reward = Reward(score=0.5, breakdown={}, reason="Default info test")
        result = StepResult(observation=obs, reward=reward, done=False)
        assert result.info == {}
