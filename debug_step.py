import requests
import json

BASE = "http://localhost:7860"

# Reset first
r = requests.post(f"{BASE}/reset", json={"task_id": "task_easy"})
session_id = r.json()["session_id"]
print(f"Session: {session_id}")

# Take a step and print FULL raw response
r = requests.post(f"{BASE}/step", json={"session_id": session_id, "action": {"action_type": "no_op"}})
print("\nFull step response:")
print(json.dumps(r.json(), indent=2))
