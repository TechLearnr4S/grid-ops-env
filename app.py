"""
app.py — FastAPI server for grid-ops-env.

Endpoints:
  POST /reset          — Start a new episode
  POST /step           — Execute one action
  GET  /state          — Get current environment state
  GET  /tasks          — List all available tasks
  POST /grade          — Grade a completed episode
  GET  /health         — Health check

Sessions are stored in memory, capped at 100 (LRU-eviction of oldest).
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from env.grid_env import GridOpsEnv
from env.models import Action, ActionType
from graders import grade as run_grader


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="grid-ops-env",
    version="1.0.0",
    description=(
        "Power Grid Emergency Operations Center — "
        "AI agent manages electricity supply, handles faults, prevents blackouts."
    ),
)

# -- CORS (all origins) ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store (max 100 sessions, evict oldest on overflow)
# ---------------------------------------------------------------------------

MAX_SESSIONS = 100
_sessions: OrderedDict[str, GridOpsEnv] = OrderedDict()


def _get_env(session_id: str) -> GridOpsEnv:
    """Look up a session by ID or raise 404."""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call POST /reset to create a session.",
        )
    # Move to end to mark as recently used
    _sessions.move_to_end(session_id)
    return _sessions[session_id]


def _store_env(session_id: str, env: GridOpsEnv) -> None:
    """Store an environment, evicting the oldest session if at capacity."""
    if len(_sessions) >= MAX_SESSIONS and session_id not in _sessions:
        oldest_key, _ = next(iter(_sessions.items()))
        del _sessions[oldest_key]
    _sessions[session_id] = env
    _sessions.move_to_end(session_id)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: str = Field(
        default="task_easy",
        description="Task scenario to initialise. One of: task_easy, task_medium, task_hard.",
    )


class ResetResponse(BaseModel):
    session_id: str
    task_id: str
    step_result: Dict[str, Any]


class StepRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by /reset")
    action: Dict[str, Any] = Field(
        ...,
        description="Action dict. Must contain 'action_type' plus any type-specific fields.",
    )


class GradeRequest(BaseModel):
    session_id: str = Field(..., description="Session ID of the episode to grade")


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    task_count = len(GridOpsEnv.list_tasks())
    print(f"GridOpsEnv server running — {task_count} tasks available")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", summary="Root ping endpoint")
async def root() -> JSONResponse:
    """Return 200 OK for automated ping."""
    return JSONResponse(
        content={"status": "ok", "message": "GridOpsEnv is running, please use /health"}
    )


@app.post("/reset", summary="Start a new episode")
async def reset(body: Optional[ResetRequest] = None) -> JSONResponse:
    """
    Create a new GridOpsEnv session for the requested task and reset it.

    Returns StepResult JSON + session_id.
    """
    if body is None:
        body = ResetRequest(task_id="task_easy")
        
    try:
        env = GridOpsEnv(body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    step_result = env.reset()
    session_id = str(uuid.uuid4())
    _store_env(session_id, env)

    return JSONResponse(
        content={
            "session_id": session_id,
            "task_id": body.task_id,
            "step_result": step_result.model_dump(),
        }
    )


@app.post("/step", summary="Execute one action in the environment")
async def step(body: StepRequest) -> JSONResponse:
    """
    Look up the session, parse the action, and advance the environment by one step.

    Returns StepResult JSON.
    """
    env = _get_env(body.session_id)

    # Parse the action dict into the Action Pydantic model
    try:
        action = Action(**body.action)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action payload: {exc}",
        )

    step_result = env.step(action)

    return JSONResponse(
        content={
            "session_id": body.session_id,
            "step_result": step_result.model_dump(),
        }
    )


@app.get("/state", summary="Get current environment state")
async def get_state(session_id: str) -> JSONResponse:
    """
    Return the full internal state of the environment as a JSON-serialisable dict.

    Includes: task_id, current_step, done, grid_state, action_history, state_history.
    """
    env = _get_env(session_id)
    return JSONResponse(content=env.state())


@app.get("/tasks", summary="List all available task scenarios")
async def list_tasks() -> JSONResponse:
    """Return metadata for all three task scenarios."""
    return JSONResponse(content=GridOpsEnv.list_tasks())


@app.post("/grade", summary="Grade a completed episode")
async def grade_episode(body: GradeRequest) -> JSONResponse:
    """
    Grade the episode identified by session_id using the appropriate task grader.

    The grader receives the final grid_state plus the full state_history
    (injected under the key _state_history) so that history-sensitive criteria
    (e.g. hospital blackout across all steps) can be evaluated correctly.

    Returns: {"score": float, "breakdown": dict, "reason": str}
    """
    env = _get_env(body.session_id)
    env_snapshot = env.state()

    task_id = env_snapshot["task_id"]
    action_history = env_snapshot["action_history"]

    # Inject state_history into the grid_state dict for history-aware graders
    grid_state = env_snapshot["grid_state"]
    grid_state["_state_history"] = env_snapshot.get("state_history", [])

    try:
        result = run_grader(task_id, grid_state, action_history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return JSONResponse(
        content={
            "session_id": body.session_id,
            "task_id": task_id,
            "steps_taken": env_snapshot["current_step"],
            "grader_result": result,
        }
    )


@app.get("/health", summary="Health check")
async def health() -> JSONResponse:
    """Return service health status."""
    return JSONResponse(
        content={
            "status": "ok",
            "version": "1.0.0",
            "environment": "grid-ops-env",
            "active_sessions": len(_sessions),
            "max_sessions": MAX_SESSIONS,
        }
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
