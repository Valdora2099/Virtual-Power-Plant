# models.py
"""
Pydantic data models for the VPP Environment.

These form the typed "contract" between the environment server and any agent.
All models are serialisable to JSON and validated on every request.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Static registry
# ---------------------------------------------------------------------------

class BatteryAsset(BaseModel):
    """Physical specification of one home battery unit (read-only)."""

    asset_id: str = Field(..., description="Unique home identifier, e.g. 'home-042'.")
    capacity_kwh: float = Field(..., description="Maximum energy storage in kWh (e.g. 13.5).")
    max_power_kw: float = Field(..., description="Maximum charge/discharge rate in kW (e.g. 5.0).")
    efficiency_rt: float = Field(
        0.90,
        description="Round-trip efficiency. 0.90 → 10 % energy lost as heat on the charging leg.",
    )


# ---------------------------------------------------------------------------
# Dynamic telemetry
# ---------------------------------------------------------------------------

class BatteryTelemetry(BaseModel):
    """Real-time snapshot of one home battery asset."""

    asset_id: str = Field(..., description="Matches a BatteryAsset.asset_id.")
    soc: float = Field(
        ..., ge=0.0, le=1.0,
        description="State of Charge: 0.0 = empty, 1.0 = full.",
    )
    current_house_load_kw: float = Field(
        ..., description="Instantaneous household power consumption in kW."
    )
    current_solar_gen_kw: float = Field(
        ..., description="Instantaneous solar panel output in kW."
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class VppAction(Action):
    """
    Dispatch command sent by the agent to all 100 home batteries simultaneously.

    The agent controls two parameters per step:

    global_charge_rate : float  in [-1.0, 1.0]
        Scaled battery charge/discharge command applied to every home.
        +1.0 → charge at full rate (buy from grid, fill battery)
        -1.0 → discharge at full rate (sell to grid, drain battery)
         0.0 → idle (no grid transaction)

    min_reserve_pct : float  in [0.0, 1.0]
        Minimum State-of-Charge the agent promises to maintain.
        Dropping below this triggers a safety violation penalty.
        Recommended: ≥ 0.15 to keep lights on at night.
    """

    global_charge_rate: float = Field(
        ...,
        ge=-1.0, le=1.0,
        description="Dispatch command: -1 = full sell, 0 = idle, +1 = full buy.",
    )
    min_reserve_pct: float = Field(
        0.2,
        ge=0.0, le=1.0,
        description="Safety floor for SoC. Agent is penalised if any battery drops below this.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class VppObservation(Observation):
    """
    Everything the agent is allowed to observe at each step.

    Note: short_term_*_forecast fields include Gaussian noise to simulate
    real-world forecast uncertainty.  The full 24-hour arrays are the
    true underlying curves (used for look-ahead planning).
    """

    timestamp: datetime = Field(..., description="Wall-clock time of this step (UTC).")
    step_id: int = Field(..., description="Step index 0–95 (15-min intervals).")
    telemetry: List[BatteryTelemetry] = Field(
        ..., description="Per-home real-time snapshot (100 entries)."
    )
    grid_frequency_hz: float = Field(
        50.0, description="Grid frequency. < 49.8 Hz = emergency requiring immediate discharge."
    )
    grid_voltage_v: float = Field(
        230.0, description="Grid voltage. > 250 V requires charging to absorb excess power."
    )
    market_price_per_mwh: float = Field(
        ..., description="Current wholesale electricity price in USD/MWh."
    )

    # Full-horizon forecasts (ground-truth — useful for planning)
    forecast_24h_price: List[float] = Field(
        ..., description="True price curve for all 96 steps (USD/MWh)."
    )
    forecast_24h_solar: List[float] = Field(
        ..., description="True solar generation curve for all 96 steps (kW/home)."
    )

    # Short-term noisy forecasts (next 4 steps = next 60 minutes)
    short_term_price_forecast: List[float] = Field(
        default_factory=list,
        description="Noisy price forecast for the next 4 steps (Gaussian σ=2.5 USD/MWh).",
    )
    short_term_solar_forecast: List[float] = Field(
        default_factory=list,
        description="Noisy solar forecast for the next 4 steps (Gaussian σ=0.25 kW).",
    )


# ---------------------------------------------------------------------------
# State (ground truth — hidden from agent in competitive evaluation)
# ---------------------------------------------------------------------------

class VppState(State):
    """
    Full ground-truth state of the VPP episode.

    Returned by GET /state.  Agents should not use this during evaluation;
    it exists for debugging, analysis, and the grader.
    """

    # ── Temporal ───────────────────────────────────────────────────────────
    current_step: int = Field(..., description="Current 15-min interval index (0–95).")
    task_tier: str = Field(
        ...,
        description="Active scenario ID: 'easy-arbitrage' | 'medium-forecast-error' | 'hard-frequency-response'.",
    )
    done: bool = Field(False, description="True when the episode has ended.")

    # ── Financial accumulators ──────────────────────────────────────────────
    cumulative_revenue_usd: float = Field(0.0, description="Total revenue from grid sales (USD).")
    cumulative_cost_usd: float = Field(0.0, description="Total cost of grid purchases (USD).")
    cumulative_profit_usd: float = Field(0.0, description="Revenue − Cost (USD).")

    # ── Safety & performance ───────────────────────────────────────────────
    blackout_events_count: int = Field(
        0, description="Number of steps where a battery hit 0 % while demand was non-zero."
    )
    safety_violations_count: int = Field(
        0, description="Cumulative count of per-home reserve violations."
    )
    grid_emergencies_ignored: int = Field(
        0, description="Steps where grid frequency < 49.8 Hz but agent was not discharging."
    )

    # ── Physical ground truth ──────────────────────────────────────────────
    actual_weather_mode: str = Field(
        ..., description="'clear_sky' | 'heatwave' | 'partly_cloudy'."
    )
    battery_true_soc: Dict[str, float] = Field(
        ..., description="Precise SoC for every home (0.0–1.0)."
    )