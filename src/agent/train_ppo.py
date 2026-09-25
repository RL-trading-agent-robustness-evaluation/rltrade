"""PPO training harness adhering to Regulation V1 and Proposal §4.2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.agent.evaluate import evaluate_agent, verify_state_responsiveness
from src.env.trading_env import RLTradingEnv


def train_ppo_agent(
    train_csv: str = "data/processed/train.csv",
    val_csv: str = "data/processed/val.csv",
    total_timesteps: int = 50_000,
    seed: int = 42,
    log_dir: str = "experiments/ppo_runs",
    model_save_path: str = "models/ppo_victim_v1",
) -> PPO:
    """Trains a PPO trading victim agent on the training split."""
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"Initializing Training Environment with {len(train_df)} rows...")
    train_env = RLTradingEnv(train_df, seed=seed)
    val_env = RLTradingEnv(val_df, seed=seed)

    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(model_save_path).parent
    models_dir.mkdir(parents=True, exist_ok=True)

    # Policy configuration
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.ReLU,
    )

    # PPO hyperparameters with entropy regularizer to prevent action collapse
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,  # Encourages state exploration and avoids policy degeneracy
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(out_dir / "tensorboard"),
    )

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(models_dir / "best_model"),
        log_path=str(out_dir / "eval_results"),
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    print(f"Starting PPO training for {total_timesteps} timesteps (seed={seed})...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    save_file = f"{model_save_path}_seed{seed}.zip"
    model.save(save_file)
    print(f"Model saved to {save_file}")

    # Evaluate on validation split
    print("\n--- Validation Evaluation ---")
    val_metrics = evaluate_agent(model, val_env, deterministic=True, save_ledger_csv=out_dir / f"val_ledger_seed{seed}.csv")
    print(f"Validation Total Return: {val_metrics['total_return_pct']:.2f}%")
    print(f"Validation Annual Return: {val_metrics['annual_return_pct']:.2f}%")
    print(f"Validation Annual Sharpe: {val_metrics['annual_sharpe']:.2f}")
    print(f"Validation Max Drawdown: {val_metrics['max_drawdown_pct']:.2f}%")
    print(f"Validation Action Distribution: {val_metrics['action_distribution']}")

    # State-responsiveness verification (Victim Gate V1)
    resp = verify_state_responsiveness(model, val_df, deterministic=True)
    print(f"State Responsiveness Passed: {resp['is_state_responsive']} (Diff Rate: {resp['action_diff_rate']*100:.1f}%)")
    print(f"Normal Actions: {resp['actions_normal_dist']}")
    print(f"Masked Actions: {resp['actions_zeroed_dist']}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Trading Victim V1")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_ppo_agent(total_timesteps=args.timesteps, seed=args.seed)
