"""Quick Phase 3 verification — no API key required."""
from __future__ import annotations
import sys, json

print("=== 1. inference.py imports clean ===")
import importlib
inf = importlib.import_module("inference")
assert hasattr(inf, "build_system_prompt")
assert hasattr(inf, "build_user_prompt")
assert hasattr(inf, "parse_action")
assert hasattr(inf, "run_episode")
assert hasattr(inf, "main")
print("  All functions present.")

print("\n=== 2. build_system_prompt() ===")
sp = inf.build_system_prompt()
assert len(sp) > 500, "System prompt too short"
for kw in ["action_type", "dispatch_generator", "shed_load", "acknowledge_alert",
           "no_op", "PRIORITY", "CRITICAL", "JSON"]:
    assert kw in sp, f"Missing keyword in system prompt: {kw}"
print(f"  System prompt: {len(sp)} chars, all required keywords present.")

print("\n=== 3. build_user_prompt() ===")
from env.grid_env import GridOpsEnv
from env.models import Action, ActionType

env = GridOpsEnv("task_easy")
sr = env.reset()
obs = sr.observation.model_dump()
up = inf.build_user_prompt(1, obs, [])
assert "STEP 1" in up
assert "GENERATORS" in up
assert "LOAD ZONES" in up
assert "ALERTS" in up
assert "WEATHER" in up
print(f"  User prompt: {len(up)} chars, all sections present.")

print("\n=== 4. parse_action — all valid action types ===")
from env.models import ActionType

test_cases = [
    ('{"action_type": "no_op"}', "no_op"),
    ('{"action_type": "dispatch_generator", "generator_id": "gen-gas-1", "target_mw": 300}', "dispatch_generator"),
    ('{"action_type": "shed_load", "zone_id": "zone-industrial", "shed_mw": 50}', "shed_load"),
    ('{"action_type": "reroute_line", "line_id": "line-3", "target_zone_id": "zone-city-south"}', "reroute_line"),
    ('{"action_type": "request_emergency_reserve", "reserve_mw": 100}', "request_emergency_reserve"),
    ('{"action_type": "send_field_crew", "line_id": "line-2", "crew_id": "crew-1"}', "send_field_crew"),
    ('{"action_type": "acknowledge_alert", "alert_id": "alert-easy-1"}', "acknowledge_alert"),
    ('{"action_type": "issue_public_notice", "notice_text": "Test notice"}', "issue_public_notice"),
]

for raw_json, expected_type in test_cases:
    action = inf.parse_action(raw_json)
    assert action.action_type.value == expected_type, f"Expected {expected_type}, got {action.action_type.value}"
    print(f"  OK: {expected_type}")

print("\n=== 5. parse_action — crash-proof on bad inputs ===")
bad_inputs = [
    "",
    "not json at all",
    '{"action_type": "explode_reactor"}',  # invalid type
    "```json\n{\"action_type\": \"no_op\"}\n```",  # markdown block
    '{"action_type": "no_op" invalid}',  # malformed JSON
    "null",
    "{}",
    "   ",
    "The grid is stable, I recommend a no_op.",  # natural language
]

for bad in bad_inputs:
    action = inf.parse_action(bad)
    assert action.action_type is not None, f"parse_action returned None for: {bad!r}"
    assert isinstance(action, Action), f"parse_action did not return Action for: {bad!r}"
    bad_repr = repr(bad)[:40]
    print(f"  OK (no crash): {bad_repr} -> {action.action_type.value}")

print("\n=== 6. parse_action handles markdown code blocks ===")
md_wrapped = '```json\n{"action_type": "dispatch_generator", "generator_id": "gen-gas-1", "target_mw": 250}\n```'
action = inf.parse_action(md_wrapped)
assert action.action_type.value == "dispatch_generator"
assert action.generator_id == "gen-gas-1"
assert action.target_mw == 250.0
print("  Markdown code block parsed correctly.")

print("\n=== 7. Dockerfile exists and has required lines ===")
import pathlib
dockerfile = pathlib.Path("Dockerfile").read_text(encoding="utf-8")
for required in ["python:3.11-slim", "WORKDIR /app", "EXPOSE 7860", "uvicorn", "app:app", "HEALTHCHECK", "useradd"]:
    assert required in dockerfile, f"Missing in Dockerfile: {required}"
print("  Dockerfile structure verified.")

print("\n=== 8. .env.example exists ===")
env_example = pathlib.Path(".env.example").read_text(encoding="utf-8")
assert "OPENAI_API_KEY" in env_example
assert "API_BASE_URL" in env_example
assert "MODEL_NAME" in env_example
assert "HF_TOKEN" in env_example
print("  .env.example has all required variables.")

print("\n=== 9. README.md is comprehensive ===")
readme = pathlib.Path("README.md").read_text(encoding="utf-8")
required_sections = [
    "GridOpsEnv", "Quick Start", "Action Space", "Observation Space",
    "Reward", "API Reference", "POST /reset", "GET /health",
    "Docker", "Environment Variables", "Baseline Scores", "Project Structure",
    "License",
]
for section in required_sections:
    assert section in readme, f"Missing section in README: {section}"
print(f"  README: {len(readme)} chars, all {len(required_sections)} required sections present.")

print("\n=== 10. Full Phase 1 test suite still passes ===")
import subprocess
result = subprocess.run(
    ["python", "-m", "pytest", "tests/test_models.py", "-q"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"  FAILED:\n{result.stdout}\n{result.stderr}")
    sys.exit(1)
lines = [l for l in result.stdout.splitlines() if l.strip()]
print(f"  {lines[-1]}")

print("\nAll Phase 3 checks passed!")
