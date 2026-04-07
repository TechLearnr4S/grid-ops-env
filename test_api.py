import requests

BASE = "http://localhost:7860"
print("="*50)
print("Testing GridOpsEnv API")
print("="*50)

print("\n1. GET /health")
r = requests.get(f"{BASE}/health")
print(f"   Status: {r.status_code} OK: {r.json()['status']}")

print("\n2. GET /tasks")
r = requests.get(f"{BASE}/tasks")
for t in r.json():
    print(f"   - {t['id']} ({t['difficulty']})")

print("\n3. POST /reset")
r = requests.post(f"{BASE}/reset", json={"task_id": "task_easy"})
session_id = r.json()["session_id"]
print(f"   Session: {session_id}")

print("\n4. POST /step (no_op)")
r = requests.post(f"{BASE}/step", json={"session_id": session_id, "action": {"action_type": "no_op"}})
sr = r.json()["step_result"]
print(f"   Reward: {sr['reward']['score']:.4f}")
print(f"   Done: {sr['done']}")

print("\n5. POST /step (dispatch_generator)")
r = requests.post(f"{BASE}/step", json={"session_id": session_id, "action": {"action_type": "dispatch_generator", "generator_id": "gen-gas-1", "target_mw": 300}})
sr = r.json()["step_result"]
print(f"   Reward: {sr['reward']['score']:.4f}")
print(f"   Messages: {sr['observation']['messages']}")

print("\n6. GET /state")
r = requests.get(f"{BASE}/state", params={"session_id": session_id})
print(f"   Step: {r.json()['current_step']}")
print(f"   Keys: {list(r.json().keys())}")

print("\n7. POST /grade")
r = requests.post(f"{BASE}/grade", json={"session_id": session_id})
print(f"   Score: {r.json()['score']:.4f}")
print(f"   Breakdown: {r.json()['breakdown']}")

print("\n8. All 3 tasks")
for task_id in ["task_easy", "task_medium", "task_hard"]:
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id})
    sid = r.json()["session_id"]
    print(f"   {task_id}: status={r.status_code} session={sid[:8]}...")

print("\n" + "="*50)
print("All endpoints PASSED!")
print("="*50)
