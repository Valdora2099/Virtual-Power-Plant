# baseline_inference.py
"""
Baseline inference script — required by the OpenEnv spec.

Runs a zero-shot LLM agent against all 3 tasks and writes baseline_scores.json.

API key priority (as per OpenEnv spec):
  1. OPENAI_API_KEY  → uses api.openai.com (GPT-4o-mini by default)
  2. GROQ_API_KEY    → falls back to api.groq.com (Llama-3 8B instant)

Environment variables:
  VPP_SERVER_URL   URL of the running FastAPI server (default: http://localhost:8000)
  OPENAI_API_KEY   OpenAI API key (preferred by OpenEnv spec)
  GROQ_API_KEY     Groq API key (fallback)
  MODEL_NAME       Override model name (optional)

Usage:
  export OPENAI_API_KEY=sk-...
  python baseline_inference.py
"""

import json
import os
import re
import sys

import requests
from openai import OpenAI

from models import VppAction

# ---------------------------------------------------------------------------
# API client setup (OpenAI first, Groq fallback)
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


if GROQ_API_KEY:
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    DEFAULT_MODEL = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    print(f"Using Groq API (fallback) with model: {DEFAULT_MODEL}")
else:
    print("ERROR: Set OPENAI_API_KEY or GROQ_API_KEY before running.", file=sys.stderr)
    sys.exit(1)

SERVER_URL = os.getenv("VPP_SERVER_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# LLM action generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert energy trader managing a Virtual Power Plant (VPP).
You must maximise profit by deciding when to buy energy from the grid (store in batteries)
and when to sell energy back to the grid (discharge batteries).

Rules:
- global_charge_rate > 0  → buy from grid (costs money, charges batteries)
- global_charge_rate < 0  → sell to grid (earns money, drains batteries)
- If grid_frequency_hz < 49.8, you MUST set a negative charge_rate (emergency discharge)
- Keep min_reserve_pct >= 0.15 to avoid blackouts at night

Respond ONLY with a valid JSON object — no explanation, no markdown fences."""

ACTION_PROMPT = """Current VPP observation:

Step: {step_id}/47  ({time_of_day})
Current price: ${price:.2f}/MWh
Grid frequency: {freq:.2f} Hz
Mean battery SoC: {mean_soc:.1%}
Min battery SoC:  {min_soc:.1%}
Mean solar output: {solar:.2f} kW/home
Mean demand:       {demand:.2f} kW/home
Next 4-step price forecast: {price_forecast}
Next 4-step solar forecast: {solar_forecast}

Task: {task_id}

Decide your action. Return JSON:
{{"global_charge_rate": <float -1.0 to 1.0>, "min_reserve_pct": <float 0.0 to 1.0>}}"""


def _summarise_obs(obs: dict) -> dict:
    """Extract key summary stats from a raw observation dict."""
    telemetry = obs.get("telemetry", [])
    socs = [t["soc"] for t in telemetry] if telemetry else [0.5]
    solar = [t["current_solar_gen_kw"] for t in telemetry] if telemetry else [0.0]
    demand = [t["current_house_load_kw"] for t in telemetry] if telemetry else [0.0]

    step = obs.get("step_id", 0)
    hour = (step * 15) // 60
    minute = (step * 15) % 60
    time_of_day = f"{hour:02d}:{minute:02d}"

    price_fc = obs.get("short_term_price_forecast") or obs.get("forecast_24h_price", [])[:4]
    solar_fc = obs.get("short_term_solar_forecast") or obs.get("forecast_24h_solar", [])[:4]

    return {
        "step_id": step,
        "time_of_day": time_of_day,
        "price": obs.get("market_price_per_mwh", 50.0),
        "freq": obs.get("grid_frequency_hz", 50.0),
        "mean_soc": sum(socs) / len(socs),
        "min_soc": min(socs),
        "solar": sum(solar) / max(len(solar), 1),
        "demand": sum(demand) / max(len(demand), 1),
        "price_forecast": [round(p, 1) for p in price_fc[:4]],
        "solar_forecast": [round(s, 2) for s in solar_fc[:4]],
    }


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM output, handling markdown fences."""
    # Strip code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


def get_llm_action(obs: dict, task_id: str) -> VppAction:
    """Query the LLM for an action given the current observation."""
    summary = _summarise_obs(obs)
    prompt = ACTION_PROMPT.format(task_id=task_id, **summary)

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,     # low temperature → more deterministic
            max_tokens=80,
        )
        text = response.choices[0].message.content.strip()
        decision = _extract_json(text)
        action = VppAction(**decision)
        print(
            f"  step={summary['step_id']:02d} t={summary['time_of_day']}"
            f"  price=${summary['price']:.0f}"
            f"  soc={summary['mean_soc']:.0%}"
            f"  → rate={action.global_charge_rate:+.2f}  reserve={action.min_reserve_pct:.2f}"
        )
        return action
    except Exception as e:
        print(f"  LLM error ({e}) — using idle action.")
        return VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, session: requests.Session) -> dict:
    """Run one full episode and return score + metadata."""
    print(f"\n{'─'*50}")
    print(f"  Task: {task_id}")
    print(f"{'─'*50}")

    # Reset
    resp = session.post(f"{SERVER_URL}/reset", params={"task_id": task_id})
    if resp.status_code != 200:
        print(f"  Reset failed: {resp.text}")
        return {"score": 0.0, "total_reward": 0.0, "steps": 0, "error": resp.text}

    obs = resp.json()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = get_llm_action(obs, task_id)
        resp = session.post(
            f"{SERVER_URL}/step",
            json={
                "global_charge_rate": action.global_charge_rate,
                "min_reserve_pct": action.min_reserve_pct,
            },
        )
        if resp.status_code != 200:
            print(f"  Step failed: {resp.text}")
            break

        data = resp.json()
        obs = data["observation"]
        total_reward += float(data["reward"])
        done = data["done"]
        steps += 1

    # Fetch grader score
    grader_resp = session.get(f"{SERVER_URL}/grader")
    grader_data = grader_resp.json()
    score = grader_data.get("score", 0.0)

    print(f"\n  Steps completed   : {steps}")
    print(f"  Total SB3 reward  : {total_reward:.2f}")
    print(f"  Profit (USD)      : ${grader_data.get('cumulative_profit_usd', 0.0):.2f}")
    print(f"  Safety violations : {grader_data.get('safety_violations', 0)}")
    print(f"  Grader score      : {score:.4f}")

    return {
        "score": score,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "profit_usd": grader_data.get("cumulative_profit_usd", 0.0),
        "safety_violations": grader_data.get("safety_violations", 0),
        "model": DEFAULT_MODEL,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"VPP Baseline Inference")
    print(f"Server : {SERVER_URL}")
    print(f"Model  : {DEFAULT_MODEL}")

    # Verify server is alive
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        assert health.status_code == 200
    except Exception as e:
        print(f"\nERROR: Cannot reach server at {SERVER_URL}: {e}")
        sys.exit(1)

    session = requests.Session()
    tasks = ["easy-arbitrage", "medium-forecast-error", "hard-frequency-response"]
    results = {}

    for task in tasks:
        results[task] = run_task(task, session)

    # Summary
    print(f"\n{'='*50}")
    print("  FINAL BASELINE SCORES")
    print(f"{'='*50}")
    for task, data in results.items():
        print(f"  {task:<32}  score={data['score']:.4f}")

    # Write baseline_scores.json (also read by GET /baseline)
    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → baseline_scores.json")


if __name__ == "__main__":
    main()