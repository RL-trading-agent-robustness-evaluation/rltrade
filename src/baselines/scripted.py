"""Scripted baselines and Gate E1 sanity validator per Regulation V1 §18 and Proposal §4.3."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd

from src.env.trading_env import RLTradingEnv


def calculate_metrics(daily_navs: List[float], costs: float, initial_nav: float = 1_000_000.0) -> Dict[str, float]:
    """Computes standardized performance and risk metrics."""
    navs = np.array(daily_navs, dtype=np.float64)
    t = len(navs)
    if t < 2:
        return {}

    total_return = (navs[-1] / initial_nav) - 1.0
    ann_return = ((navs[-1] / initial_nav) ** (252.0 / t)) - 1.0 if navs[-1] > 0 else -1.0

    # Daily percentage returns
    daily_rets = np.diff(navs) / navs[:-1]
    mean_ret = np.mean(daily_rets)
    std_ret = np.std(daily_rets) + 1e-12
    ann_sharpe = (mean_ret / std_ret) * np.sqrt(252.0)

    # Max Drawdown
    peaks = np.maximum.accumulate(navs)
    drawdowns = (peaks - navs) / peaks
    max_dd = float(np.max(drawdowns))

    return {
        "final_nav": float(navs[-1]),
        "total_return_pct": total_return * 100.0,
        "annual_return_pct": ann_return * 100.0,
        "annual_vol_pct": float(std_ret * np.sqrt(252.0) * 100.0),
        "annual_sharpe": float(ann_sharpe),
        "max_drawdown_pct": max_dd * 100.0,
        "total_cost": float(costs),
    }


def run_scripted_policy(
    env: RLTradingEnv,
    policy_fn: Callable[[int, np.ndarray, Dict[str, Any]], int],
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes a policy function throughout the entire dataset."""
    obs, info = env.reset(seed=seed)
    daily_navs = [env.ledger.state.nav]
    action_counts = {0: 0, 1: 0, 2: 0}
    total_cost = 0.0

    step_idx = 0
    while True:
        action = policy_fn(step_idx, obs, info)
        action_counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        daily_navs.append(info["nav_close"])
        total_cost += info["transaction_cost"]
        step_idx += 1
        if terminated or truncated:
            break

    metrics = calculate_metrics(daily_navs, costs=total_cost, initial_nav=env.initial_cash)
    metrics["action_distribution"] = {k: v / step_idx for k, v in action_counts.items()}
    return metrics


def run_gate_e1_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Runs all 5 Gate E1 baselines and returns summary DataFrame."""
    results = {}

    # 1. All Cash (0%)
    env_cash = RLTradingEnv(df)
    results["All Cash (0%)"] = run_scripted_policy(env_cash, lambda step, obs, info: 0)

    # 2. Buy & Hold (100%)
    env_bh = RLTradingEnv(df)
    results["Buy & Hold (100%)"] = run_scripted_policy(env_bh, lambda step, obs, info: 2)

    # 3. Static 50/50
    env_50 = RLTradingEnv(df)
    results["Static 50/50"] = run_scripted_policy(env_50, lambda step, obs, info: 1)

    # 4. Alternating 0% / 100%
    env_alt = RLTradingEnv(df)
    results["Alternating (0/100)"] = run_scripted_policy(env_alt, lambda step, obs, info: 2 if step % 2 == 0 else 0)

    # 5. Moving Average Crossover (20 SMA > 50 SMA -> 100%, else 0%)
    # Compute causal SMAs from close prices
    close = df["close"].to_numpy()
    sma20 = pd.Series(close).rolling(20).mean().to_numpy()
    sma50 = pd.Series(close).rolling(50).mean().to_numpy()
    
    def ma_policy(step, obs, info):
        if step < 50:
            return 0
        return 2 if sma20[step] > sma50[step] else 0

    env_ma = RLTradingEnv(df)
    results["MA Crossover (20/50)"] = run_scripted_policy(env_ma, ma_policy)

    # 6. Random Agent (Must have Sharpe ~ 0 / negative with costs)
    rng = np.random.RandomState(42)
    env_rand = RLTradingEnv(df)
    results["Random Agent"] = run_scripted_policy(env_rand, lambda step, obs, info: rng.choice([0, 1, 2]))

    report_df = pd.DataFrame(results).T
    return report_df


if __name__ == "__main__":
    from src.data.adapter import build_canonical_dataset
    df = build_canonical_dataset()
    report = run_gate_e1_baselines(df)
    print("\n=== Gate E1 Scripted Baselines Performance ===")
    print(report[["total_return_pct", "annual_return_pct", "annual_vol_pct", "annual_sharpe", "max_drawdown_pct", "total_cost"]])
