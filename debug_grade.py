import requests
import json

BASE = "http://localhost:7860"

r = requests.post(f"{BASE}/reset", json={"task_id": "task_easy"})
session_id = r.json()["session_id"]

r = requests.post(f"{BASE}/grade", json={"session_id": session_id})
print("Grade response:")
print(json.dumps(r.json(), indent=2))
