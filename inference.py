import os
import re
import json
import time
import textwrap
from typing import List

from openai import OpenAI

from env.grid_env import GridOpsEnv
from env.models import Action
from graders import grade

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

API_KEY = HF_TOKEN
TEMPERATURE = 0.2
MAX_TOKENS = 512
MAX_STEPS = 20
FALLBACK_ACTION = "no_op"

# ---------------------------------------------------------------------------
# Client — deferred init so the module is importable without env vars set
# (tests and validators import inference.py without an API key)
# ---------------------------------------------------------------------------

client: OpenAI = None  # type: ignore[assignment]  # initialised in main() / run_episode()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a power grid operations AI.
    Your job: prevent blackouts, maintain grid stability, manage budget.

    Available action_types and their required JSON fields:
      dispatch_generator        -> requires: generator_id, target_mw
      shed_load                 -> requires: zone_id, amount_mw
      reroute_line              -> requires: line_id, target_line_id
      request_emergency_reserve -> requires: amount_mw
      send_field_crew           -> requires: line_id
      acknowledge_alert         -> requires: alert_id
      issue_public_notice       -> requires: zone_id, message
      no_op                     -> no extra fields

    Priority order: critical > high > medium > low zones.

    Reply with ONLY a valid JSON object. No explanation. No markdown. Example:
    {"action_type": "dispatch_generator", "generator_id": "plant_gas_A", "target_mw": 300}

    If unsure, reply with: {"action_type": "no_op"}
""").strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])


def build_user_prompt(step: int, observation, history: List[str]) -> str:
    gs = observation.grid_state

    generators_lines = "\n".join(
        f"  {g.id}: status={g.status.value}, "
        f"output={g.current_mw:.0f}/{g.max_mw:.0f} MW, "
        f"cost=${g.cost_per_mw:.1f}/MW"
        for g in gs.generators
    ) or "  None"

    tx_lines = "\n".join(
        f"  {ln.id}: status={ln.status.value}, "
        f"flow={ln.current_mw:.0f}/{ln.max_mw:.0f} MW"
        for ln in gs.lines
    ) or "  None"

    zone_lines = "\n".join(
        f"  {z.id}: priority={z.priority.value}, "
        f"demand={z.demand_mw:.0f} MW, served={z.served_mw:.0f} MW, "
        f"blacked_out={z.is_blacked_out}"
        for z in gs.load_zones
    ) or "  None"

    unacked = [a for a in gs.alerts if not a.acknowledged]
    if unacked:
        alert_lines = "\n".join(
            f"  {a.id}: [{a.severity.value.upper()}] {a.message}"
            for a in unacked
        )
    else:
        alert_lines = "  None"

    weather = gs.weather

    return textwrap.dedent(f"""
        Step: {step}
        Goal: {observation.goal}
        Current Time (hour): {gs.current_hour}
        Budget Remaining: ${gs.budget_remaining:.0f}

        --- GENERATORS ---
        {generators_lines}

        --- TRANSMISSION LINES ---
        {tx_lines}

        --- LOAD ZONES ---
        {zone_lines}

        --- ACTIVE ALERTS ---
        {alert_lines}

        --- WEATHER ---
          wind_speed_kph={weather.wind_speed_kph:.0f}, storm_active={weather.storm_active},
          wind_forecast_mw={weather.wind_forecast_mw:.0f}, solar_forecast_mw={weather.solar_forecast_mw:.0f}

        Previous steps:
        {build_history_lines(history)}

        Reply with exactly one JSON action object.
    """).strip()


# ---------------------------------------------------------------------------
# Action parser — must never raise an exception
# ---------------------------------------------------------------------------


def parse_model_action(response_text: str) -> Action:
    if not response_text:
        return Action(action_type="no_op")
    # Strip markdown fences
    cleaned = re.sub(r"```json|```", "", response_text).strip()
    # Find first JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return Action(**data)
        except Exception:
            return Action(action_type="no_op")
    return Action(action_type="no_op")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(task_id: str, client: OpenAI) -> dict:
    env = GridOpsEnv(task_id)
    history: List[str] = []
    history_dicts: List[dict] = []
    step_rewards: List[str] = []
    reward = 0.0

    print(f"[START] task={task_id} env=grid-ops-env model={MODEL_NAME}")

    try:
        result = env.reset()
        observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step, observation, history)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                response_text = ""

            action = parse_model_action(response_text)
            result = env.step(action)
            observation = result.observation

            reward = result.reward.score if result.reward else 0.0
            history_line = f"Step {step}: {action.action_type} -> reward {reward:+.2f}"
            history.append(history_line)
            history_dicts.append(action.model_dump())

            reward_str = f"{reward:.2f}"
            step_rewards.append(reward_str)
            done_str = "true" if result.done else "false"
            action_str = json.dumps(action.model_dump(), separators=(',', ':'))
            error_str = "null"
            if hasattr(result, "info") and result.info.get("error"):
                error_str = str(result.info["error"]).replace("\\n", " ")

            print(f"[STEP] step={step} action={action_str} reward={reward_str} done={done_str} error={error_str}")

            if result.done:
                break

        else:
            pass

    finally:
        pass

    grade_result = grade(task_id, env.state(), history_dicts)
    success_str = "true" if grade_result["score"] > 0.0 else "false"
    
    print(f"[END] success={success_str} steps={len(history)} rewards={','.join(step_rewards)}")

    return {
        "task_id": task_id,
        "steps_taken": len(history),
        "final_reward": reward if history else 0.0,
        "grade_result": grade_result,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = {}

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        result = run_episode(task_id, client)
        results[task_id] = result

    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
