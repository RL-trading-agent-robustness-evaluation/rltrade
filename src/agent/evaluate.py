"""Evaluation module and state-responsiveness verifier per Regulation V1 §13, §20."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from src.env.trading_env import RLTradingEnv
from src.baselines.scripted import calculate_metrics


def evaluate_agent(
    model: BaseAlgorithm,
    env: RLTradingEnv,
    deterministic: bool = True,
    save_ledger_csv: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Rolls out an agent, records full step ledger, and computes metrics."""
    obs, info = env.reset(seed=42)
    ledger_records: List[Dict[str, Any]] = []
    daily_navs = [env.ledger.state.nav]
    action_counts = {0: 0, 1: 0, 2: 0}
    total_cost = 0.0

    step_idx = 0
    while True:
        action, _states = model.predict(obs, deterministic=deterministic)
        action = int(action)
        action_counts[action] += 1

        obs, reward, terminated, truncated, step_info = env.step(action)
        daily_navs.append(step_info["nav_close"])
        total_cost += step_info["transaction_cost"]
        ledger_records.append(step_info)

        step_idx += 1
        if terminated or truncated:
            break

    metrics = calculate_metrics(daily_navs, costs=total_cost, initial_nav=env.initial_cash)
    metrics["action_distribution"] = {k: v / step_idx for k, v in action_counts.items()}
    metrics["total_steps"] = step_idx

    # Export audit ledger if requested
    if save_ledger_csv is not None:
        out_path = Path(save_ledger_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_df = pd.DataFrame(ledger_records)
        ledger_df.to_csv(out_path, index=False)
        print(f"Step ledger written to {out_path} ({len(ledger_df)} steps)")

    return metrics


def verify_state_responsiveness(
    model: BaseAlgorithm,
    df: pd.DataFrame,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Runs Victim Gate V1 state-responsiveness check (Regulation §20).
    
    Verifies that zeroing out or perturbing observation features measurably changes action distribution.
    A degenerate agent that always outputs a fixed action will show 0 difference.
    """
    env_normal = RLTradingEnv(df)
    obs_normal, _ = env_normal.reset(seed=42)
    actions_normal = []

    while True:
        act, _ = model.predict(obs_normal, deterministic=deterministic)
        actions_normal.append(int(act))
        obs_normal, _, term, trunc, _ = env_normal.step(int(act))
        if term or trunc:
            break

    # Run with zeroed-out market features (dims 0..14)
    env_zeroed = RLTradingEnv(df)
    obs_z, _ = env_zeroed.reset(seed=42)
    actions_zeroed = []

    while True:
        # Zero out market dims, preserve internal dims (dims 15, 16)
        obs_z_masked = obs_z.copy()
        obs_z_masked[:15] = 0.0
        act_z, _ = model.predict(obs_z_masked, deterministic=deterministic)
        actions_zeroed.append(int(act_z))
        obs_z, _, term, trunc, _ = env_zeroed.step(int(act_z))
        if term or trunc:
            break

    n = min(len(actions_normal), len(actions_zeroed))
    diff_count = sum(1 for i in range(n) if actions_normal[i] != actions_zeroed[i])
    action_diff_rate = diff_count / n if n > 0 else 0.0

    is_responsive = action_diff_rate > 0.05
    return {
        "is_state_responsive": is_responsive,
        "action_diff_rate": action_diff_rate,
        "actions_normal_dist": {k: actions_normal.count(k) / len(actions_normal) for k in [0, 1, 2]},
        "actions_zeroed_dist": {k: actions_zeroed.count(k) / len(actions_zeroed) for k in [0, 1, 2]},
    }


if __name__ == "__main__":
    import argparse
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(description="Evaluate Trained Trading Agent")
    parser.add_argument("--model", type=str, default="models/ppo_victim_v1_seed42.zip", help="Path to saved model zip")
    parser.add_argument("--data", type=str, default="data/processed/val.csv", help="Path to dataset CSV to evaluate on")
    parser.add_argument("--output-ledger", type=str, default="experiments/ppo_runs/eval_ledger.csv", help="Path to save step ledger CSV")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    data_df = pd.read_csv(args.data)
    model = PPO.load(model_path)
    env = RLTradingEnv(data_df)

    print(f"\nEvaluating {model_path.name} on {args.data} ({len(data_df)} days)...")
    metrics = evaluate_agent(model, env, deterministic=True, save_ledger_csv=args.output_ledger)

    print("\n================ FINANCIAL PERFORMANCE REPORT ================")
    print(f"Total Return:         {metrics['total_return_pct']:>8.2f}%")
    print(f"Annualized Return:    {metrics['annual_return_pct']:>8.2f}%")
    print(f"Annualized Volatility:{metrics['annual_vol_pct']:>8.2f}%")
    print(f"Annualized Sharpe:    {metrics['annual_sharpe']:>8.2f}")
    print(f"Maximum Drawdown:     {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"Total Transaction Cost:  ${metrics['total_cost']:>10.2f}")
    print(f"Action Distribution:  Cash={metrics['action_distribution'][0]*100:.1f}%, 50%={metrics['action_distribution'][1]*100:.1f}%, 100%={metrics['action_distribution'][2]*100:.1f}%")
    print("==============================================================")

    resp = verify_state_responsiveness(model, data_df, deterministic=True)
    print(f"\nVictim Gate V1 (State-Responsiveness): {'PASSED' if resp['is_state_responsive'] else 'FAILED'}")
    print(f"Decision Difference when Observation Masked: {resp['action_diff_rate']*100:.1f}%\n")
