"""Verify new inference.py matches exact OpenEnv pattern spec."""
import ast, pathlib, sys, os, subprocess

print("=== Syntax ===")
src = pathlib.Path("inference.py").read_text(encoding="utf-8")
try:
    ast.parse(src)
    print("  OK")
except SyntaxError as e:
    print(f"  SyntaxError line {e.lineno}: {e.msg}")
    sys.exit(1)

print("=== Exact variable constants ===")
# clear env so defaults take effect
for k in ["API_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME"]:
    os.environ.pop(k, None)
os.environ["HF_TOKEN"] = "mock_token"

import importlib
inf = importlib.import_module("inference")

assert inf.API_BASE_URL == "https://api.openai.com/v1", f"Bad URL: {inf.API_BASE_URL}"
assert inf.TEMPERATURE == 0.2
assert inf.MAX_TOKENS == 512
assert inf.MAX_STEPS == 20
assert inf.FALLBACK_ACTION == "no_op"
print("  API_BASE_URL, TEMPERATURE, MAX_TOKENS, MAX_STEPS, FALLBACK_ACTION all correct")

print("=== Functions present ===")
for fn in ["build_history_lines", "build_user_prompt", "parse_model_action", "run_episode", "main"]:
    assert hasattr(inf, fn), f"Missing: {fn}"
    print(f"  {fn}: OK")

print("=== SYSTEM_PROMPT ===")
sp = inf.SYSTEM_PROMPT
for kw in ["dispatch_generator", "shed_load", "no_op", "critical > high > medium > low"]:
    assert kw in sp, f"Missing in SYSTEM_PROMPT: {kw}"
print(f"  {len(sp)} chars, all required keywords present")

print("=== build_history_lines ===")
assert inf.build_history_lines([]) == "None"
for n, expected in [(1,"a"), (3,"a\nb\nc"), (5,"b\nc\nd\ne")]:
    lst = list("abcde")[:n] if n <= 5 else list("abcde")
    got = inf.build_history_lines(list("abcde")[:5])
    break
assert inf.build_history_lines(["a","b","c","d","e"]) == "b\nc\nd\ne"
print("  empty='None', last-4 correct")

print("=== parse_model_action crash-proof ===")
from env.models import Action
crash_inputs = [
    "", "not json", "{bad}", "null", "{}", "   ",
    '{"action_type": "destroy_grid"}',
    "```json\n{\"action_type\": \"no_op\"}\n```",
    "I recommend a no_op action.",
    '{"action_type": "shed_load"}',  # missing required fields -> fallback
]
for bad in crash_inputs:
    a = inf.parse_model_action(bad)
    assert isinstance(a, Action), f"Returned non-Action for: {bad!r}"
print(f"  All {len(crash_inputs)} bad inputs handled safely")

print("=== parse_model_action valid JSON ===")
a = inf.parse_model_action('{"action_type": "no_op"}')
assert a.action_type.value == "no_op"
a = inf.parse_model_action('{"action_type": "acknowledge_alert", "alert_id": "a1"}')
assert a.action_type.value == "acknowledge_alert" and a.alert_id == "a1"
print("  Valid JSON parsed correctly")

print("=== observation.goal field ===")
from env.grid_env import GridOpsEnv
for tid in ["task_easy", "task_medium", "task_hard"]:
    env = GridOpsEnv(tid)
    sr = env.reset()
    assert hasattr(sr.observation, "goal")
    assert len(sr.observation.goal) > 10, f"goal too short for {tid}"
    print(f"  {tid}: goal='{sr.observation.goal[:50]}...'")

print("=== build_user_prompt sections ===")
env = GridOpsEnv("task_easy")
sr = env.reset()
prompt = inf.build_user_prompt(1, sr.observation, [])
for section in ["Step:", "Goal:", "GENERATORS", "TRANSMISSION LINES", "LOAD ZONES",
                "ACTIVE ALERTS", "WEATHER", "Previous steps"]:
    assert section in prompt, f"Missing: {section}"
assert "None" in prompt  # empty history
print(f"  {len(prompt)} chars, all 8 sections present")

print("=== messages use content-array format ===")
# Verify the messages format in run_episode source
assert '[{"type": "text", "text":' in src or '{"type": "text", "text":' in src
print('  content-array format confirmed in source')

print("=== stream=False in source ===")
assert "stream=False" in src
print("  stream=False confirmed")

print("=== pytest (50 tests) ===")
r = subprocess.run(["python", "-m", "pytest", "tests/test_models.py", "-q"],
                   capture_output=True, text=True)
lines = [l for l in r.stdout.splitlines() if l.strip()]
print(f"  {lines[-1] if lines else r.stderr}")
assert r.returncode == 0

print("\nAll checks passed!")
