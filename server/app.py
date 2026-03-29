# server/app.py
"""
FastAPI application exposing the VPP environment via the OpenEnv HTTP interface.

Endpoints
---------
POST /reset          Start a new episode
POST /step           Take one action
GET  /state          Return current ground-truth state
GET  /tasks          List available tasks + action schema
GET  /grader         Return episode score (0.0–1.0)
GET  /baseline       Return pre-computed baseline LLM scores
GET  /health         Liveness probe (used by Docker HEALTHCHECK)
"""

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from models import VppAction, VppObservation, VppState
from server.vpp_environment import VppEnvironment


# ---------------------------------------------------------------------------
# Lifespan: create a warm environment instance on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    env = VppEnvironment()
    yield


app = FastAPI(
    title="VPP Orchestrator — OpenEnv",
    description=(
        "Virtual Power Plant environment: manage 100 home batteries "
        "to maximise grid profit while maintaining safety constraints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

env: VppEnvironment | None = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_env() -> VppEnvironment:
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised — call /reset first.")
    return env


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Docker HEALTHCHECK liveness probe."""
    return {"status": "ok", "env_ready": env is not None}


@app.post("/reset")
async def reset(task_id: str = Query(..., description="Task ID: easy-arbitrage | medium-forecast-error | hard-frequency-response")):
    """
    Reset the environment and start a new episode.

    Returns the initial observation.
    """
    global env
    if env is None:
        env = VppEnvironment()

    valid_tasks = {"easy-arbitrage", "medium-forecast-error", "hard-frequency-response"}
    if task_id not in valid_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_id '{task_id}'. Valid: {sorted(valid_tasks)}",
        )

    obs = env.reset(task_id)
    return obs


@app.post("/step")
async def step(action: VppAction):
    """
    Take one action in the environment.

    Returns observation, reward, done flag, and diagnostic info.
    """
    e = _require_env()
    if e.state is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    if e.state.done:
        raise HTTPException(status_code=400, detail="Episode finished — call /reset to start a new one.")

    obs, reward, done, info = e.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def get_state():
    """Return the current ground-truth state (hidden from agent in production)."""
    e = _require_env()
    return e.state


@app.get("/tasks")
async def get_tasks():
    """
    List all available tasks and the action schema.

    Used by the OpenEnv validator and by agent frameworks to discover
    what actions are valid.
    """
    return {
        "tasks": [
            {
                "id": "easy-arbitrage",
                "description": (
                    "High solar production, low household demand, flat $50/MWh price. "
                    "Strategy: sell solar surplus. Profit target: $500."
                ),
                "difficulty": "easy",
                "profit_target_usd": 500.0,
            },
            {
                "id": "medium-forecast-error",
                "description": (
                    "Heatwave event: AC demand spikes 4× from 10:00–14:00. "
                    "Sinusoidal pricing rewards time-of-use arbitrage. "
                    "Agent must manage forecast uncertainty. Profit target: $200."
                ),
                "difficulty": "medium",
                "profit_target_usd": 200.0,
            },
            {
                "id": "hard-frequency-response",
                "description": (
                    "Grid stress: a single-step 10× price spike at 12:30 (step 50). "
                    "Grid frequency drops to 49.5 Hz — agent must discharge immediately. "
                    "If batteries are depleted before the spike, revenue is lost. "
                    "Requires look-ahead planning and reserve management. Profit target: $1000."
                ),
                "difficulty": "hard",
                "profit_target_usd": 1000.0,
            },
        ],
        "action_schema": VppAction.model_json_schema(),
        "observation_schema": VppObservation.model_json_schema(),
    }


@app.get("/grader")
async def get_grader_score():
    """
    Return the deterministic grader score for the completed (or in-progress) episode.

    Score is in [0.0, 1.0]:
      1.0 = profit target met, zero safety violations
      0.0 = no profit or extreme violation count
    """
    if env is None or env.state is None:
        return {"score": 0.0, "detail": "No episode in progress."}

    score = env.get_current_task_score()
    state = env.state
    return {
        "score": score,
        "cumulative_profit_usd": state.cumulative_profit_usd,
        "safety_violations": state.safety_violations_count,
        "grid_emergencies_ignored": state.grid_emergencies_ignored,
        "steps_completed": state.current_step,
        "done": state.done,
    }


@app.get("/baseline")
async def get_baseline():
    """
    Return pre-computed baseline scores produced by baseline_inference.py.

    These scores represent a Llama-3 LLM agent playing each task with a
    zero-shot prompt — no fine-tuning, no RL.
    """
    baseline_path = os.path.join(os.path.dirname(__file__), "..", "baseline_scores.json")
    try:
        with open(baseline_path, "r") as f:
            scores = json.load(f)
        return scores
    except FileNotFoundError:
        # Fallback stubs so the endpoint always returns valid JSON
        return {
            "easy-arbitrage": {"score": 0.0, "note": "Run baseline_inference.py to generate scores."},
            "medium-forecast-error": {"score": 0.0, "note": "Run baseline_inference.py to generate scores."},
            "hard-frequency-response": {"score": 0.0, "note": "Run baseline_inference.py to generate scores."},
        }