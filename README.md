# Virtual Power Plant Orchestrator

> **OpenEnv environment** — an AI agent manages 100 home batteries to maximise grid profit while keeping every household's lights on.

---

## Motivation

Renewable energy is intermittent. Solar panels generate power only when the sun shines, but cities need electricity 24 hours a day. In 2026, **Virtual Power Plants (VPPs)** solve this by aggregating thousands of home batteries into a single, grid-scale asset that can be charged when energy is cheap and discharged when it is scarce.

This environment places an AI agent in the role of a VPP operator managing a neighbourhood of **100 homes**, each with a **13.5 kWh battery** and a **5 kW solar panel**. The agent must:

- Decide every 15 minutes whether to **buy** energy from the grid (charge batteries), **sell** energy back (discharge), or **idle**.
- Maximise financial profit over a 24-hour episode.
- Respect a hard safety constraint: no home's battery may drop below a user-defined reserve level.
- Respond instantly to grid emergencies (frequency events).

This is a real control problem faced by companies like Tesla Energy, Sonnen, and OhmConnect every day.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://huggingface.co/spaces/<your-space>/vpp-env
cd vpp-env
pip install -r requirements.txt
```

### 2. Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Interact via HTTP

```bash
# Reset
curl -X POST "http://localhost:8000/reset?task_id=easy-arbitrage"

# Step
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{"global_charge_rate": -0.8, "min_reserve_pct": 0.2}'

# Grader
curl "http://localhost:8000/grader"
```

### 4. Docker

```bash
docker build -t vpp-env .
docker run -p 7860:7860 vpp-env
```

---

## Environment Design

### Episode structure

| Property | Value |
|---|---|
| Step duration | 15 minutes |
| Episode length | 96 steps (24 hours) |
| Assets | 100 home batteries |
| Battery capacity | 13.5 kWh each |
| Max charge/discharge rate | 5.0 kW each |
| Round-trip efficiency | 90 % |
| Starting SoC | 50 % (all homes) |

### Physics

Each step, for every home battery:

```
effective_charge_kw = charge_kw × η   (if charging)  else  charge_kw
delta_kwh           = (solar - demand + effective_charge_kw) × 0.25
new_soc             = old_soc + delta_kwh / capacity_kwh
grid_profit         = -charge_kw × 0.25 × (price_USD/MWh / 1000)
```

Profit is positive when selling (`charge_kw < 0`), negative when buying.

---

## Action Space

```json
{
  "global_charge_rate": -0.8,
  "min_reserve_pct": 0.2
}
```

| Field | Type | Range | Description |
|---|---|---|---|
| `global_charge_rate` | float | [-1.0, +1.0] | +1 = buy at full rate, -1 = sell at full rate, 0 = idle |
| `min_reserve_pct` | float | [0.0, 1.0] | Safety floor. Violations below this are penalised. |

---

## Observation Space

```json
{
  "timestamp": "2026-03-28T06:00:00Z",
  "step_id": 24,
  "telemetry": [
    { "asset_id": "home-000", "soc": 0.52, "current_house_load_kw": 0.3, "current_solar_gen_kw": 1.2 },
    "... (100 entries)"
  ],
  "grid_frequency_hz": 50.0,
  "grid_voltage_v": 230.0,
  "market_price_per_mwh": 52.3,
  "forecast_24h_price": [...],
  "forecast_24h_solar": [...],
  "short_term_price_forecast": [52.1, 54.0, 55.3, 53.8],
  "short_term_solar_forecast": [1.2, 1.8, 2.3, 2.9]
}
```

Key signals the agent should use:

- `market_price_per_mwh` — sell when high, buy when low
- `grid_frequency_hz` — if < 49.8 Hz, **discharge immediately** (grid emergency)
- `telemetry[*].soc` — track battery state to avoid blackouts
- `forecast_24h_price` — plan ahead for the 10× price spike (hard task)
- `short_term_*_forecast` — noisy 60-minute look-ahead (realistic forecast error)

---

## Tasks

### Easy — Arbitrage (`easy-arbitrage`)

**Scenario:** Clear sky, low household demand, flat $50/MWh electricity price.

**Strategy:** Solar charges the batteries during the day. Sell surplus to the grid whenever the battery is above the reserve level.

**Profit target:** $500  
**Difficulty:** ⭐☆☆

---

### Medium — Forecast Error (`medium-forecast-error`)

**Scenario:** A heatwave hits. Air conditioning demand spikes **4×** between 10:00–14:00 (steps 40–56). Electricity price is sinusoidal ($40–$60/MWh).

**Strategy:** The agent must anticipate the demand spike from the forecast and reserve enough battery capacity to keep homes cool, while still profiting from time-of-use price arbitrage.

**Profit target:** $200  
**Difficulty:** ⭐⭐☆

---

### Hard — Frequency Response (`hard-frequency-response`)

**Scenario:** The grid experiences stress at **12:30 (step 50)**. For exactly one 15-minute interval, the grid price spikes to **10× normal** ($500/MWh) and the frequency drops to **49.5 Hz** (below the 49.8 Hz emergency threshold).

**Challenge:** To capture the $500/MWh payout, the agent must have batteries charged and ready. But the sun is reduced (0.7× solar) and base demand is high (1.2×), so batteries drain faster earlier in the day. A greedy agent that sells all morning will have empty batteries when the spike arrives.

**Profit target:** $1000  
**Difficulty:** ⭐⭐⭐

---

## Reward Function

```
reward = step_profit
       − reserve_violations × 0.05     (per home, per step)
       − emergency_penalty             (2.0 if freq < 49.8 and not discharging)
```

The reward function provides **dense, gradient signal** at every step — not a sparse binary outcome. This makes the environment suitable for RL algorithms (PPO, SAC) that require smooth value estimation.

---

## Grader (0.0 → 1.0)

```
profit_ratio      = min(1.0, cumulative_profit / target_profit)
violation_penalty = min(0.40, safety_violations × 0.05)
emergency_penalty = min(0.30, grid_emergencies_ignored × 0.10)

score = max(0.0, profit_ratio − violation_penalty − emergency_penalty)
```

The grader is **fully deterministic and programmatic** — no LLM-as-judge. Same agent on same task will always produce the same score because both the environment curves and forecast noise are seeded by `task_id`.

---

## Baseline Scores

Run the baseline inference script to generate scores:

```bash
export OPENAI_API_KEY=sk-...   # or GROQ_API_KEY
python baseline_inference.py
```

Pre-computed results (zero-shot GPT-4o-mini, temperature=0.1):

| Task | Score | Profit (USD) | Violations |
|---|---|---|---|
| easy-arbitrage | 0.71 | $357 | 0 |
| medium-forecast-error | 0.43 | $118 | 3 |
| hard-frequency-response | 0.12 | $190 | 8 |

*Scores above represent the LLM baseline without any fine-tuning or RL.*

---

## RL Training

A PPO agent using curriculum learning outperforms the LLM baseline significantly:

```bash
# Start the server first
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Train (3-phase curriculum: easy → medium → hard)
python train_rl.py
```

Training uses `stable-baselines3` with a 3-phase curriculum:

| Phase | Task | Steps | Purpose |
|---|---|---|---|
| 1 | easy-arbitrage | 200,000 | Learn buy-low/sell-high |
| 2 | medium-forecast-error | 150,000 | Handle demand uncertainty |
| 3 | hard-frequency-response | 150,000 | Learn reserve strategy |

Monitor training: `tensorboard --logdir ./vpp_tensorboard/`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/tasks` | List tasks + action schema |
| POST | `/reset?task_id=...` | Start new episode |
| POST | `/step` | Take one action |
| GET | `/state` | Ground-truth state (debugging) |
| GET | `/grader` | Episode score 0.0–1.0 |
| GET | `/baseline` | Pre-computed LLM baseline scores |

---

## Project Structure

```
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI application
│   ├── vpp_environment.py   # Core simulation engine
│   └── task_curves.py       # Deterministic solar/demand/price curves
├── models.py                # Pydantic schemas (Action, Observation, State)
├── client.py                # OpenEnv EnvClient wrapper
├── gymwrapper.py            # Gymnasium wrapper for RL training
├── train_rl.py              # PPO curriculum training script
├── baseline_inference.py    # LLM baseline script (OpenEnv required)
├── baseline_scores.json     # Pre-computed baseline results
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # Container definition
└── requirements.txt
```

---

## Requirements

```
openenv-core[core]>=0.2.1
fastapi>=0.115.0
uvicorn>=0.24.0
openai>=1.0.0
numpy>=1.24.0
requests>=2.31.0
pydantic>=2.0.0
gymnasium>=0.29.0
stable-baselines3>=2.0.0
```