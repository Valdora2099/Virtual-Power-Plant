# server/task_curves.py
"""
Deterministic 48-step energy curves for each task tier.

Episode design
──────────────
  One step  = 15 minutes
  48 steps  = 12 hours  (06:00 → 18:00)
  Step 0    = 06:00     Step 47 = 17:45

Time index reference
  Step  0 → 06:00      Step 16 → 10:00
  Step  8 → 08:00      Step 24 → 12:00
  Step 12 → 09:00      Step 26 → 12:30  ← grid stress spike (hard task)
  Step 14 → 09:30      Step 32 → 14:00  ← heatwave ends (medium task)
  Step 16 → 10:00      Step 47 → 17:45

All functions take a full task_id string (e.g. 'easy-arbitrage') and
extract the tier internally — no caller needs to pre-strip the suffix.
"""

import numpy as np

EPISODE_STEPS = 48          # 12-hour episode
GRID_STRESS_STEP = 26       # 12:30 — single-step 10× price spike (hard task)
HEATWAVE_START   = 16       # 10:00
HEATWAVE_END     = 32       # 14:00  (exclusive)


def _tier(task_id: str) -> str:
    """'easy-arbitrage' → 'easy', etc."""
    return task_id.split("-")[0].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Solar
# ─────────────────────────────────────────────────────────────────────────────

def solar_curve(task_id: str) -> np.ndarray:
    """
    48-step solar generation curve (kW per home).

    Shape: bell curve centred on solar noon (step 24 = 12:00).
    Window spans steps 0–47 (06:00–17:45), so the full arc is visible.

    Easy   → abundant sun  (1.5×, peak ~6.0 kW)
    Medium → normal sun    (1.0×, peak ~4.0 kW)
    Hard   → reduced sun   (0.7×, peak ~2.8 kW) — forces careful reserve mgmt
    """
    steps = np.arange(EPISODE_STEPS)
    # Bell curve: zero at step 0 (06:00), peak at step 24 (12:00), zero again at step 48
    base = np.maximum(0.0, 4.0 * np.sin(np.pi * steps / EPISODE_STEPS))

    multipliers = {"easy": 1.5, "medium": 1.0, "hard": 0.7}
    base *= multipliers.get(_tier(task_id), 1.0)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Demand
# ─────────────────────────────────────────────────────────────────────────────

def demand_curve(task_id: str) -> np.ndarray:
    """
    48-step household demand curve (kW per home).

    Base shape: morning ramp (peaks ~09:00, step 12) + afternoon plateau.

    Easy   → low demand    (0.5×)
    Medium → heatwave AC spike steps 16–31 (10:00–13:45), 4× demand
    Hard   → high demand   (1.2×)
    """
    steps = np.arange(EPISODE_STEPS)

    # Gaussian morning peak at step 12 (09:00) + flat afternoon
    morning_peak = 0.5 * np.exp(-0.5 * ((steps - 12) / 6) ** 2)
    base = 0.25 + morning_peak
    base = np.clip(base, 0.15, 1.2)

    tier = _tier(task_id)
    if tier == "easy":
        base *= 0.5
    elif tier == "medium":
        heatwave = np.ones(EPISODE_STEPS)
        heatwave[HEATWAVE_START:HEATWAVE_END] = 4.0
        base *= heatwave
    elif tier == "hard":
        base *= 1.2

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Price
# ─────────────────────────────────────────────────────────────────────────────

def price_curve(task_id: str) -> np.ndarray:
    """
    48-step wholesale electricity price (USD/MWh).

    Easy   → flat $50/MWh  (no arbitrage skill required)
    Medium → sinusoidal $35–$65/MWh  (time-of-use arbitrage rewarded)
    Hard   → sinusoidal + single-step 10× spike at step 26 (12:30)
              Agent must keep batteries charged to capitalise on the spike.
    """
    steps = np.arange(EPISODE_STEPS)
    tier  = _tier(task_id)

    if tier == "easy":
        base = np.full(EPISODE_STEPS, 50.0)
    else:
        # Morning cheap, midday peak, afternoon moderate
        base = 50.0 + 15.0 * np.sin(2 * np.pi * (steps - 8) / EPISODE_STEPS)
        base = np.clip(base, 30.0, 70.0)

    if tier == "hard":
        base[GRID_STRESS_STEP] *= 10.0   # one-step spike at 12:30

    return base