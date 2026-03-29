# train_rl.py
"""
Curriculum RL training for the VPP environment using Stable-Baselines3 PPO.

Training strategy (3-phase curriculum):
  Phase 1 (easy-arbitrage,    200k steps) — learn basic buy-low-sell-high
  Phase 2 (medium-forecast-error, 150k steps) — transfer + handle heatwave
  Phase 3 (hard-frequency-response, 150k steps) — fine-tune for spike event

Total: 500k environment steps.  Estimated wall time: ~20 min on a modern CPU.

Usage:
  # Start the VPP server first:
  uvicorn server.app:app --host 0.0.0.0 --port 8000

  # Then run:
  python train_rl.py
"""

import os
import time
import requests

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from gymwrapper import VppGymEnv

SERVER_URL = os.getenv("VPP_SERVER_URL", "http://localhost:8000")
TENSORBOARD_LOG = "./vpp_tensorboard/"
CHECKPOINT_DIR = "./checkpoints/"


# ---------------------------------------------------------------------------
# Factory — required by DummyVecEnv to avoid the lambda closure trap
# ---------------------------------------------------------------------------

def make_env(task_id: str):
    """Return a callable that creates a fresh VppGymEnv (no live HTTP at call time)."""
    def _init():
        e = VppGymEnv(base_url=SERVER_URL, task_id=task_id)
        return Monitor(e)   # wraps for episodic reward/length logging
    return _init


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model: PPO, task_id: str, n_episodes: int = 3) -> dict:
    """
    Run n_episodes deterministically and return mean SB3 reward + grader score.

    The grader score (0–1) is fetched from /grader after each episode and averaged.
    """
    rewards = []
    grader_scores = []

    for ep in range(n_episodes):
        env = VppGymEnv(base_url=SERVER_URL, task_id=task_id)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += float(reward)
            if truncated:
                break

        env.close()

        # Fetch grader score from server (reads internal state of the env instance)
        try:
            resp = requests.get(f"{SERVER_URL}/grader", timeout=5)
            grader_scores.append(resp.json().get("score", 0.0))
        except Exception:
            grader_scores.append(0.0)

        rewards.append(ep_reward)

    return {
        "task_id": task_id,
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "mean_grader_score": round(float(np.mean(grader_scores)), 4),
    }


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def train_phase(
    model: PPO,
    task_id: str,
    total_timesteps: int,
    phase_name: str,
) -> PPO:
    """Train (or fine-tune) `model` on `task_id` for `total_timesteps` steps."""
    print(f"\n{'='*60}")
    print(f"  Phase: {phase_name}  |  Task: {task_id}  |  Steps: {total_timesteps:,}")
    print(f"{'='*60}")

    train_env = DummyVecEnv([make_env(task_id)])
    train_env = VecMonitor(train_env)

    model.set_env(train_env)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, total_timesteps // 5),   # 5 checkpoints per phase
        save_path=os.path.join(CHECKPOINT_DIR, task_id),
        name_prefix=f"vpp_ppo_{task_id}",
        verbose=0,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        reset_num_timesteps=False,   # keep global step counter across phases
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"  Phase complete in {elapsed/60:.1f} min.")

    train_env.close()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    # ── Sanity check: is the server reachable? ────────────────────────────
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        assert resp.status_code == 200
        print(f"Server reachable at {SERVER_URL}.")
    except Exception as e:
        print(f"ERROR: Cannot reach VPP server at {SERVER_URL}.\n"
              f"Start it with:  uvicorn server.app:app --host 0.0.0.0 --port 8000\n"
              f"Details: {e}")
        return

    # ── Phase 1: Initialise model on easy task ────────────────────────────
    init_env = DummyVecEnv([make_env("easy-arbitrage")])
    init_env = VecMonitor(init_env)

    model = PPO(
        "MlpPolicy",
        init_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=TENSORBOARD_LOG,
        policy_kwargs={"net_arch": [256, 256]},   # larger network for complex task
    )
    init_env.close()

    model = train_phase(model, "easy-arbitrage",          200_000, "Phase 1 — Easy")
    model = train_phase(model, "medium-forecast-error",   150_000, "Phase 2 — Medium")
    model = train_phase(model, "hard-frequency-response", 150_000, "Phase 3 — Hard")

    # ── Save final model ──────────────────────────────────────────────────
    model.save("vpp_ppo_final")
    print("\nFinal model saved → vpp_ppo_final.zip")

    # ── Final evaluation across all tasks ─────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL EVALUATION")
    print("="*60)
    results = []
    for task in ["easy-arbitrage", "medium-forecast-error", "hard-frequency-response"]:
        r = evaluate(model, task, n_episodes=3)
        results.append(r)
        print(
            f"  {r['task_id']:<32}"
            f"  reward={r['mean_reward']:>8.2f} ± {r['std_reward']:<7.2f}"
            f"  grader={r['mean_grader_score']:.4f}"
        )

    print("\nTensorBoard logs: ./vpp_tensorboard/")
    print("Checkpoints:      ./checkpoints/")
    print("To view training: tensorboard --logdir ./vpp_tensorboard/")


if __name__ == "__main__":
    main()