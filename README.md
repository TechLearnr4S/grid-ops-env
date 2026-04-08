---
title: Meta Hackathon Grid Ops
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---
# ⚡ GridOpsEnv
### Power Grid Emergency Operations — OpenEnv Environment

![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)
![Python 3.11](https://img.shields.io/badge/python-3.11-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

**GridOpsEnv** places an AI agent in the role of a regional electricity grid operator during an emergency. The agent must dispatch generators, reroute power flows along transmission lines, prioritise load zones by criticality, and manage a finite budget — all while the grid evolves in real time across a simulated 24-hour demand curve.

The environment is designed to evaluate multi-step reasoning under resource constraint. Unlike toy grid problems, GridOpsEnv features compounding failures: a single unaddressed line overload can cascade into faults that black out entire districts, forcing the agent to balance urgency against budget efficiency. Observations are partially observable (wind output is uncertain during storms; fault propagation is delayed), requiring the agent to reason under uncertainty.

GridOpsEnv is the first power-grid emergency scenario in the OpenEnv ecosystem. It provides a continuous, interpretable reward signal with a clear human analogue — the fraction of demand served across zones weighted by their social priority — making agent behaviour transparent to both researchers and domain experts.

---

## Quick Start

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/grid-ops-env
cd grid-ops-env

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env
# Edit .env — set OPENAI_API_KEY

# Run baseline agent across all 3 tasks
python inference.py
```

> **Tip:** To use a HuggingFace-hosted model instead of OpenAI, set `API_BASE_URL=https://api-inference.huggingface.co/v1` and `HF_TOKEN=your_token`.

---

## Environment Description

### Grid Topology

```
                        ╔══════════════════════╗
                        ║     GENERATORS       ║
                        ║  ┌──────────────┐    ║
                        ║  │ Nuclear 500MW│    ║
                        ║  │ Gas A   300MW│    ║
                        ║  │ Gas B   200MW│    ║
                        ║  │ Wind    150MW│    ║
                        ║  │ Hydro   200MW│    ║
                        ║  └──────┬───────┘    ║
                        ╚═════════╪════════════╝
                                  │ Generation Pool
                    ┌─────────────┼─────────────┐
                    │             │             │
             ┌──────▼──────┐ ┌───▼───────┐ ┌──▼──────────┐
             │  Line 1     │ │  Line 2   │ │  Line 3     │
             │  Artery N   │ │  Feeder S │ │  Hospital   │
             │  0-600 MW   │ │  0-250 MW │ │  0-150 MW   │
             └──────┬──────┘ └───┬───────┘ └──┬──────────┘
                    │            │             │
           ┌────────▼──┐  ┌──────▼─────┐ ┌────▼──────────┐
           │ City      │  │ Industrial │ │ Hospital      │
           │ PRIORITY: │  │ PRIORITY:  │ │ PRIORITY:     │
           │   HIGH    │  │  MEDIUM    │ │  CRITICAL     │
           └───────────┘  └────────────┘ └───────────────┘
```

### Time Simulation

The environment advances one step per action. Each step increments the simulation clock by one hour, following a **sinusoidal 24-hour demand curve**:

| Hour Range | Multiplier | Description |
|---|---|---|
| 03:00–05:00 | 0.70× | Overnight trough — minimum demand |
| 08:00–12:00 | ~1.05–1.20× | Morning ramp |
| 14:00–16:00 | **1.40×** | Afternoon peak |
| 18:00–20:00 | 1.25–1.35× | Evening peak demand surge |
| 22:00–02:00 | ~0.80–0.90× | Night falloff |

### Partial Observability

- **Wind uncertainty**: During active storms, `wind_forecast_mw` overestimates actual output by up to 47%. The agent sees the forecast but not the true value.
- **Fault propagation delay**: Overloaded lines raise an alert in the step they overload, but only trip to `faulted` on the **next step** if no corrective action is taken — giving the agent one step to respond.
- **Demand curve noise**: Demand scaling applies each tick, so a line that was at 80% capacity can cross 100% on the next tick without any agent action.

---

## Action Space

All 8 action types, their required fields, and descriptions:

| `action_type` | Required Fields | Optional Fields | Description |
|---|---|---|---|
| `dispatch_generator` | `generator_id`, `target_mw` | — | Ramp a generator up/down, or bring offline/tripped units online |
| `shed_load` | `zone_id`, `shed_mw` | — | Curtail demand in a zone (reduces both served and demand) |
| `reroute_line` | `line_id`, `target_zone_id` | — | Redirect a line's power flow to a different zone |
| `request_emergency_reserve` | `reserve_mw` | `cost_override` | Purchase reserve capacity instantly at $50/MW (premium) |
| `send_field_crew` | `line_id` | `crew_id` | Dispatch repair crew to a faulted line (3 steps to repair) |
| `acknowledge_alert` | `alert_id` | — | Acknowledge an active alert (contributes 10% to reward) |
| `issue_public_notice` | `notice_text` | `affected_zone_ids` | Broadcast a public communication (required for hard task score) |
| `no_op` | — | — | Wait one step without action |

---

## Observation Space

Each step the agent receives a full `GridState` observation:

| Field | Type | Description |
|---|---|---|
| `generators` | List | Each generator: id, name, status, current\_mw, max\_mw, cost\_per\_mw, fuel\_type |
| `lines` | List | Each line: id, name, from\_zone, to\_zone, current\_mw, max\_mw, status, repair\_steps\_remaining |
| `load_zones` | List | Each zone: id, name, demand\_mw, served\_mw, priority, is\_blacked\_out |
| `alerts` | List | Active alerts with severity, message, acknowledged flag, step\_raised |
| `weather` | Object | wind\_speed\_kph, temperature\_c, storm\_active, wind\_forecast\_mw, solar\_forecast\_mw |
| `budget_remaining` | float | Remaining operational budget ($) |
| `current_step` | int | Steps elapsed in the episode |
| `current_hour` | int | Simulated hour of day (0–23) |
| `total_mw_demanded` | float | Aggregate MW demand across all zones |
| `total_mw_served` | float | Aggregate MW delivered across all zones |

---

## Tasks

### Task 1 — Peak Demand Dispatch

**Difficulty:** ⭐  
**Task ID:** `task_easy`  
**Max Steps:** 6  

**Scenario:** It is 18:00 — the evening peak. Three generators are available but only two are online, providing 600 MW against 900 MW of demand. The hydro unit is offline. One medium-severity alert warns that generation headroom is critically low.

**Initial state:**
- Nuclear: 400/500 MW (online) — baseload, cheap
- Gas Peaker: 200/300 MW (online) — can ramp to 300 MW
- Hydro Unit 1: 0/200 MW (offline) — needs to be started
- City Centre: 600 MW demand, 400 MW served (HIGH priority)
- Industrial Park: 300 MW demand, 200 MW served (MEDIUM priority)

**Success criteria:**
1. Total served / total demanded ≥ 95% in final state
2. Gas plant dispatched to higher output AND hydro unit brought online
3. All alerts acknowledged
4. Achieved within ≤ 4 steps (full efficiency score)

**Expected optimal agent sequence:**
1. `acknowledge_alert` (alert-easy-1)
2. `dispatch_generator` (gen-hydro-1, target\_mw=200)
3. `dispatch_generator` (gen-gas-1, target\_mw=300)

**Baseline score (gpt-4o-mini, temp=0):** `0.72`

---

### Task 2 — Line Fault Recovery

**Difficulty:** ⭐⭐⭐  
**Task ID:** `task_medium`  
**Max Steps:** 10  

**Scenario:** Transmission Line 2 has faulted, blacking out City South (200 MW, HIGH priority). Line 3 is the only viable bypass but is already at 85% capacity — rerouting without first shedding load will push it over 100% and trigger a cascade fault. Two alerts are active: a critical fault notification and a high-severity overload warning.

**Initial state:**
- 3 generators online, total 800 MW capacity serving 750 MW
- Line 1: healthy, 300/400 MW (75%)
- Line 2: FAULTED — carries no power, repair crew not yet dispatched
- Line 3: healthy, 212/250 MW (85%) — at risk of overload if rerouted naively
- City South: 200 MW demand, 0 MW served — BLACKED OUT
- Industrial: 250 MW demand, 250 MW served

**Success criteria:**
1. City South not blacked out in final state (0.30)
2. Field crew dispatched to Line 2 (0.20)
3. Line 3 never exceeded 100% capacity at any step (0.20)
4. Critical alert acknowledged within first 3 steps (0.15)
5. Load shedding applied to Industrial zone (not high/critical zones) (0.15)

**Expected optimal agent sequence:**
1. `acknowledge_alert` (alert-med-1) — critical alert within step 1
2. `shed_load` (zone-industrial, 50 MW) — free headroom on Line 3
3. `reroute_line` (line-3, zone-city-south) — restore City South
4. `send_field_crew` (line-2) — begin permanent repair

**Baseline score (gpt-4o-mini, temp=0):** `0.54`

---

### Task 3 — Cascading Failure

**Difficulty:** ⭐⭐⭐⭐⭐  
**Task ID:** `task_hard`  
**Max Steps:** 15  

**Scenario:** During peak evening load (20:00), Northfield Nuclear (500 MW) has tripped offline. This has pushed Lines 1 and 2 to 110% and 105% of their rated capacity — they will cascade-fault on the next tick unless immediate corrective action is taken. A severe storm is active, making wind output unreliable (actual: 80 MW vs forecast: 150 MW). Budget is critically limited at $3,000. The hospital must never lose power.

**Initial state:**
- Nuclear (500 MW): TRIPPED — 500 MW of baseload suddenly gone
- Gas Peaker A (300 MW): online at max capacity
- Gas Peaker B (200 MW): online at max capacity
- Wind Farm (150 MW rated): actually producing 80 MW (storm)
- Line 1: OVERLOADED at 110% — will trip next tick
- Line 2: OVERLOADED at 105% — will trip next tick
- Hospital: 100 MW demanded, 95 MW served (CRITICAL)
- City Centre: 400 MW demanded, 380 MW served (HIGH)
- Industrial: 300 MW demanded, 275 MW served (MEDIUM)
- Suburbs: 200 MW demanded, 140 MW served (LOW)
- Budget: $3,000 (tight)

**Success criteria:**
1. Hospital District never blacked out at ANY step (0.30) — checked across full episode history
2. Lines 1 and 2 never reach `faulted` status (0.20) — cascade prevention
3. Budget ≥ $0 in final state (0.20)
4. Total generation within ±10% of demand by step 8 (0.15)
5. `issue_public_notice` called at least once AND all critical alerts acknowledged (0.15)

**Expected agent priorities:**
1. Immediately shed industrial/suburban load to relieve Line 1 and Line 2 overload
2. Acknowledge all alerts, especially the nuclear trip
3. Issue public notice (required for communication score)
4. Consider emergency reserve for hospital protection
5. Do NOT attempt to restart nuclear — too expensive with $3,000 budget

**Baseline score (gpt-4o-mini, temp=0):** `0.31`

---

## Reward Function

The step reward is a weighted sum of four components, scored 0.0–1.0:

```
R(s, a, s') = 0.40 × MW_served_ratio
            + 0.30 × blackout_score
            + 0.20 × budget_efficiency
            + 0.10 × alert_response
```

### Multi-Objective Optimization: Reliability vs. Sustainability

GridOpsEnv (developed by **Team ImpactX**) introduces a novel reward mechanism that forces agents to balance three competing objectives:
1.  **Reliability (65%)**: Preventing blackouts and meeting load demand. 
2.  **Sustainability (15%)**: Minimizing the carbon footprint (kg CO2 per MW served) by prioritizing carbon-neutral sources like wind and nuclear.
3.  **Efficiency (20%)**: Managing a limited operational budget and responding to critical alerts promptly.

| Component | Weight | Formula | Description |
|---|---|---|---|
| `mw_served_ratio` | 0.35 | `total_mw_served / total_mw_demanded` | Fraction of total demand being met |
| `blackout_score` | 0.30 | `1.0 - (penalty / total_weight)` | Priority-weighted blackout check |
| `sustainability` | 0.15 | `1.0 - (intensity / 1000)` | Reward for carbon-neutral energy |
| `budget_efficiency` | 0.15 | `1.0 - (spend / budget) * 5` | Penalises capital depletion |
| `alert_response` | 0.05 | `acked / total_alerts` | Timely alert acknowledgement |

**Priority weights used in `blackout_score`:**
| Priority | Weight |
|---|---|
| `critical` | 1.00 |
| `high` | 0.60 |
| `medium` | 0.30 |
| `low` | 0.10 |

**Example:** If the hospital (critical) is blacked out and the city (high) is serving 100% demand, blackout penalty = 1.0 / (1.0 + 0.6 + 0.3 + 0.1) = 0.50. The `blackout_score` component = (1 - 0.50) × 0.30 = **0.15** — a significant penalty.

**Partial credit:** The grader applies partial credit for near-optimal outcomes. For example, in `grade_easy`, `supply_restored` gives full 0.35 for ≥95% served, but partial credit for 70–95% (proportional), rewarding meaningful improvement even if the full target is missed.

---

## API Reference

All endpoints run on port **7860** (HF Spaces) or **8000** (local dev).

### `POST /reset` — Start a new episode

```bash
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task_easy"}'
```

**Response:**
```json
{
  "session_id": "3f2a9b1c-...",
  "task_id": "task_easy",
  "step_result": {
    "observation": { "task_id": "task_easy", "step": 0, "grid_state": {...}, "messages": [...], "done": false },
    "reward": { "score": 0.0, "breakdown": {}, "reason": "Episode start — no reward on reset." },
    "done": false,
    "info": { "task_id": "task_easy", "max_steps": 6, "event": "reset" }
  }
}
```

---

### `POST /step` — Execute one action

```bash
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "YOUR_SESSION_ID",
       "action": {
         "action_type": "dispatch_generator",
         "generator_id": "gen-gas-1",
         "target_mw": 300
       }
     }'
```

**Response:**
```json
{
  "session_id": "YOUR_SESSION_ID",
  "step_result": {
    "observation": { "step": 1, "grid_state": {...}, "messages": ["Generator 'Riverside Gas Peaker' ramped..."], "done": false },
    "reward": { "score": 0.68, "breakdown": { "mw_served_ratio": 0.32, "blackout_score": 0.24, ... }, "reason": "..." },
    "done": false,
    "info": { "step": 1, "budget_remaining": 9200.0, "mw_served": 700.0, "mw_demanded": 900.0, "active_blackouts": 0 }
  }
}
```

---

### `GET /state` — Get current environment state

```bash
curl "http://localhost:7860/state?session_id=YOUR_SESSION_ID"
```

**Response:** Full environment snapshot including `grid_state`, `action_history`, and `state_history` (one snapshot per step).

---

### `GET /tasks` — List all available tasks

```bash
curl http://localhost:7860/tasks
```

**Response:**
```json
[
  { "id": "task_easy", "name": "Peak Demand Dispatch", "difficulty": "easy", "max_steps": 6 },
  { "id": "task_medium", "name": "Line Fault Recovery", "difficulty": "medium", "max_steps": 10 },
  { "id": "task_hard", "name": "Cascading Failure", "difficulty": "hard", "max_steps": 15 }
]
```

---

### `POST /grade` — Grade a completed episode

```bash
curl -X POST http://localhost:7860/grade \
     -H "Content-Type: application/json" \
     -d '{"session_id": "YOUR_SESSION_ID"}'
```

**Response:**
```json
{
  "session_id": "YOUR_SESSION_ID",
  "task_id": "task_easy",
  "steps_taken": 4,
  "grader_result": {
    "score": 0.85,
    "breakdown": {
      "supply_restored": 0.35,
      "correct_dispatch": 0.25,
      "alert_acknowledged": 0.15,
      "efficiency": 0.10,
      "no_unnecessary_actions": 0.0
    },
    "reason": "Supply restored: 98.3% of demand met | Correct dispatch: gas plant ramped + hydro online | ..."
  }
}
```

---

### `GET /health` — Health check

```bash
curl http://localhost:7860/health
```

**Response:**
```json
{ "status": "ok", "version": "1.0.0", "environment": "grid-ops-env", "active_sessions": 2, "max_sessions": 100 }
```

---

## Docker Deployment

### Build and run locally

```bash
docker build -t grid-ops-env .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e MODEL_NAME=gpt-4o-mini \
  grid-ops-env
```

### Verify the server is running

```bash
curl http://localhost:7860/health
# {"status":"ok","version":"1.0.0","environment":"grid-ops-env"}
```

### Run inference inside the container

```bash
docker run --rm \
  -e OPENAI_API_KEY=your_key \
  -e MODEL_NAME=gpt-4o-mini \
  grid-ops-env \
  python inference.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key for `inference.py` |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API base URL (any OpenAI-compatible endpoint) |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model to use for inference |
| `HF_TOKEN` | No | — | HuggingFace token (alternative to `OPENAI_API_KEY`) |

\* Either `OPENAI_API_KEY` or `HF_TOKEN` must be set when running `inference.py`.

---

## Baseline Scores

All baselines measured with `gpt-4o-mini` at temperature 0, single run:

| Task | Difficulty | Baseline Score | Notes |
|---|---|---|---|
| `task_easy` | Easy ⭐ | **0.72** | Model reliably dispatches gas + hydro but takes 5–6 steps |
| `task_medium` | Medium ⭐⭐⭐ | **0.54** | Crew dispatch often missed; Line 3 sometimes overloaded |
| `task_hard` | Hard ⭐⭐⭐⭐⭐ | **0.31** | Nuclear compensation rarely achieved; public notice often skipped |
| **Overall** | — | **0.52** | Average across all 3 tasks |

**Notes on baseline methodology:**
- Budget temperature (`TEMPERATURE=0.0`) for deterministic comparison
- Maximum 20 steps per episode regardless of task `max_steps`
- No few-shot examples provided — zero-shot system prompt only
- Grader scores are computed after episode completion, not per-step reward

---

## Project Structure

```
grid-ops-env/
├── env/
│   ├── __init__.py          # Package exports (all models + GridSimulator + GridOpsEnv)
│   ├── models.py            # Pydantic v2 domain models (Generator, LoadZone, Action, ...)
│   ├── grid_simulator.py    # Stateless physics engine — apply_action, tick, calculate_reward
│   └── grid_env.py          # GridOpsEnv — main environment class (reset / step / state)
├── graders/
│   ├── __init__.py          # Grader registry + grade() dispatcher
│   ├── grade_easy.py        # Deterministic grader for task_easy (5 criteria)
│   ├── grade_medium.py      # Deterministic grader for task_medium (5 criteria)
│   └── grade_hard.py        # Deterministic grader for task_hard (5 criteria)
├── tasks/
│   ├── __init__.py          # Package exports
│   └── task_definitions.py  # TASK_EASY, TASK_MEDIUM, TASK_HARD GridState configs
├── tests/
│   └── test_models.py       # 50 pytest tests for all Pydantic models
├── app.py                   # FastAPI server — 6 endpoints (reset/step/state/tasks/grade/health)
├── inference.py             # Baseline LLM agent — runs all 3 tasks, grades, saves JSON
├── Dockerfile               # Container for HF Spaces (python:3.11-slim, port 7860)
├── requirements.txt         # Pinned Python dependencies
├── openenv.yaml             # OpenEnv manifest (tasks, action/observation spaces, reward range)
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full test suite (50 tests)
pytest tests/ -v

# Run Phase 2 smoke test (environment loop + graders + FastAPI import)
python smoke_test_phase2.py
```

---

## Extending GridOpsEnv

### Adding a new task

1. Add a new config dict to `tasks/task_definitions.py` following the `TASK_EASY` pattern
2. Register it in `_registry` inside `get_task_by_id()`
3. Add a grader function in `graders/grade_newdiff.py`
4. Register the grader in `graders/__init__.py`
5. Add the task to `openenv.yaml`

### Swapping the physics model

`GridSimulator` is a pure Python class with no framework dependencies. Subclass it and override `apply_action()`, `tick()`, or `calculate_reward()` to experiment with alternative physics models.

### Using a custom LLM

Set `API_BASE_URL` to any OpenAI-compatible endpoint:

```bash
# HuggingFace Inference API
export API_BASE_URL=https://api-inference.huggingface.co/v1
export HF_TOKEN=hf_your_token
export MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
python inference.py

# Local Ollama
export API_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export MODEL_NAME=llama3.1
python inference.py
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

```
Copyright (c) 2024 grid-ops-env contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```