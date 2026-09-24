"""Gymnasium-compliant RL Trading Environment adhering strictly to Regulation V1."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.features.indicator_engine import ObservationBuilder, TOTAL_STATE_DIMS
from src.ledger.core import DoubleEntryLedger


class RLTradingEnv(gym.Env):
    """Clean single-asset RL trading victim environment per Regulation V1 [FROZEN].
    
    Action Space: Discrete(3) -> {0: 0.0, 1: 0.5, 2: 1.0} target position weights.
    Observation Space: Box(low=-inf, high=inf, shape=(17,), dtype=float32).
    Execution: Next-open execution (Decision at close t, executed at open t+1).
    Reward: Open-to-open pre-trade NAV log return log(V_{t+2}^{open,-} / V_{t+1}^{open,-}).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_cash: float = 1_000_000.0,
        cost_bps: float = 10.0,
        asset_permno: int = 84398,  # SPY
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        self.n_records = len(self.df)
        if self.n_records < 65:
            raise ValueError(f"Insufficient data rows ({self.n_records}) for 60-day feature normalization")

        self.initial_cash = initial_cash
        self.cost_bps = cost_bps
        self.asset_permno = asset_permno

        # Action space: Discrete(3) -> 0.0, 0.5, 1.0 (Regulation §6.2)
        self.action_space = spaces.Discrete(3)

        # Observation space: 15 market features + 2 internal features = 17 dims (Proposal §4.2)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(TOTAL_STATE_DIMS,),
            dtype=np.float32,
        )

        # Build causal feature engine
        self.obs_builder = ObservationBuilder(self.df, norm_window=60)
        self.ledger = DoubleEntryLedger(initial_cash=self.initial_cash, cost_bps=self.cost_bps)

        self.current_step = 0
        self.v_t1_open_pre = self.initial_cash
        self.last_action = 0

        if seed is not None:
            self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.ledger.reset()
        self.current_step = 0
        self.last_action = 0

        # Initial pre-trade open snapshot for day 1
        open_0 = float(self.df.loc[0, "open"])
        self.v_t1_open_pre = self.ledger.snapshot_pretrade_nav(open_0)

        # Initial observation at close of t=0
        obs = self.obs_builder.get_observation(
            t=0,
            current_weight=0.0,
            cash_ratio=1.0,
        )

        info = {
            "step": 0,
            "decision_date": self.df.loc[0, "date"],
            "nav": self.ledger.state.nav,
            "cash_available": self.ledger.state.cash_available,
            "shares": self.ledger.state.shares,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advances execution clock from close t to open t+1, then to open t+2 (Regulation §7)."""
        action = int(action)
        t = self.current_step
        t1 = t + 1
        t2 = t + 2

        # Step 1: Accrue interest from t to t1 on spendable cash
        cash_ret_t1 = float(self.df.loc[t, "cash_return_to_next"])
        interest_t1 = self.ledger.process_cash_interest(cash_ret_t1)

        # Step 2: Process corporate actions on day t1
        div_t1 = float(self.df.loc[t1, "dividend_amount"])
        split_t1 = float(self.df.loc[t1, "split_factor"])
        ca_t1 = self.ledger.process_corporate_actions(div_t1, split_t1, is_pay_date=True)

        # Step 3: Snapshot pre-trade NAV at t1 open (V_{t+1}^{open,-})
        open_t1 = float(self.df.loc[t1, "open"])
        v_t1_pre = self.ledger.snapshot_pretrade_nav(open_t1)

        # Step 4: Execute action at t1 open
        shares_pre = self.ledger.state.shares
        cash_pre_trade = self.ledger.state.cash_available
        trade_res = self.ledger.execute_target_weight(action_id=action, open_price=open_t1)
        v_t1_post = trade_res["nav_open_post"]

        # Step 5: Mark-to-market at t1 close
        close_t1 = float(self.df.loc[t1, "close"])
        v_t1_close = self.ledger.snapshot_close_nav(close_t1)

        # Check if t2 is within bounds
        terminated = (t2 >= self.n_records - 1)
        truncated = False

        if not terminated:
            # Step 6: Accrue interest from t1 to t2
            cash_ret_t2 = float(self.df.loc[t1, "cash_return_to_next"])
            interest_t2 = self.ledger.process_cash_interest(cash_ret_t2)

            # Step 7: Corporate actions at t2 open
            div_t2 = float(self.df.loc[t2, "dividend_amount"])
            split_t2 = float(self.df.loc[t2, "split_factor"])
            ca_t2 = self.ledger.process_corporate_actions(div_t2, split_t2, is_pay_date=True)

            # Step 8: Pre-trade NAV snapshot at t2 open (V_{t+2}^{open,-})
            open_t2 = float(self.df.loc[t2, "open"])
            v_t2_pre = self.ledger.snapshot_pretrade_nav(open_t2)

            # Compute open-to-open pre-trade NAV log return reward (Regulation §6.5)
            reward = float(math.log(max(v_t2_pre, 1e-8) / max(v_t1_pre, 1e-8)))

            # Next observation frozen at close t1
            nav_denom = max(v_t1_close, 1e-8)
            cash_ratio_t1 = float(self.ledger.state.cash_available / nav_denom)
            pos_weight_t1 = float((self.ledger.state.shares * close_t1) / nav_denom)
            obs_next = self.obs_builder.get_observation(
                t=t1,
                current_weight=pos_weight_t1,
                cash_ratio=cash_ratio_t1,
            )
        else:
            # Final terminal step
            reward = float(math.log(max(v_t1_close, 1e-8) / max(v_t1_pre, 1e-8)))
            obs_next = np.zeros(TOTAL_STATE_DIMS, dtype=np.float32)
            open_t2 = open_t1
            v_t2_pre = v_t1_close

        # Audit outputs (Regulation §13)
        info = {
            "asset_permno": self.asset_permno,
            "decision_date": self.df.loc[t, "date"],
            "execution_date": self.df.loc[t1, "date"],
            "action_id": action,
            "target_weight": trade_res["target_weight"],
            "realized_weight": trade_res["realized_weight"],
            "shares_pre": shares_pre,
            "shares_post": trade_res["shares_post"],
            "delta_shares": trade_res["delta_shares"],
            "true_open": open_t1,
            "true_close": close_t1,
            "cash_pre_trade": cash_pre_trade,
            "trade_notional": trade_res["trade_notional"],
            "transaction_cost": trade_res["transaction_cost"],
            "cash_post_trade": trade_res["cash_post_trade"],
            "market_value_close": self.ledger.state.market_value,
            "nav_close": v_t1_close,
            "nav_open_pre": v_t1_pre,
            "nav_open_post": v_t1_post,
            "nav_next_open_pre": v_t2_pre,
            "reward": reward,
            "step": t,
        }

        self.current_step += 1
        self.last_action = action
        return obs_next, reward, terminated, truncated, info
