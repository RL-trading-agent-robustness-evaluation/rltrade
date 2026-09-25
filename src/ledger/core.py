"""Strict double-entry ledger implementing Regulation V1 Part E and Part F."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class LedgerState:
    cash_available: float = 1_000_000.0
    shares: int = 0
    dividend_receivable: float = 0.0
    market_value: float = 0.0
    nav: float = 1_000_000.0
    cumulative_transaction_cost: float = 0.0
    cumulative_cash_interest: float = 0.0
    cumulative_dividends: float = 0.0

    # Snapshots
    nav_open_pre: float = 1_000_000.0
    nav_open_post: float = 1_000_000.0
    nav_close: float = 1_000_000.0


class DoubleEntryLedger:
    """Rigorous double-entry ledger tracking cash, shares, receivable, costs, and NAV invariants."""

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        cost_bps: float = 10.0,
        prototype_dividend_mode: str = "ex_date_receivable_pay_date_cash",
    ):
        self.initial_cash = float(initial_cash)
        self.cost_rate = float(cost_bps) / 10_000.0  # 10 bps = 0.001
        self.prototype_dividend_mode = prototype_dividend_mode

        self.state = LedgerState(
            cash_available=self.initial_cash,
            shares=0,
            dividend_receivable=0.0,
            market_value=0.0,
            nav=self.initial_cash,
            nav_open_pre=self.initial_cash,
            nav_open_post=self.initial_cash,
            nav_close=self.initial_cash,
        )

        self.action_weights = {0: 0.0, 1: 0.5, 2: 1.0}
        self.audit_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Resets ledger to initial conditions (Regulation §6.3)."""
        self.state = LedgerState(
            cash_available=self.initial_cash,
            shares=0,
            dividend_receivable=0.0,
            market_value=0.0,
            nav=self.initial_cash,
            nav_open_pre=self.initial_cash,
            nav_open_post=self.initial_cash,
            nav_close=self.initial_cash,
        )
        self.audit_log.clear()

    def assert_nav_identity(self, price: float, context: str = "") -> None:
        """Enforces Regulation §14 NAV numerical identity invariant:
        |NAV - (Cash + shares * price + Receivable)| <= 1e-10 * max(1, |NAV|)
        """
        exact = self.state.cash_available + float(self.state.shares) * float(price) + self.state.dividend_receivable
        diff = abs(self.state.nav - exact)
        tol = 1e-10 * max(1.0, abs(self.state.nav))
        if diff > tol:
            raise AssertionError(
                f"NAV identity violated ({context}): NAV={self.state.nav:.8f}, Exact={exact:.8f}, Diff={diff:.2e} > Tol={tol:.2e}"
            )
        assert self.state.cash_available >= -1e-8, f"Negative cash detected ({context}): {self.state.cash_available}"
        assert self.state.shares >= 0 and isinstance(self.state.shares, (int, np.integer)), f"Invalid shares: {self.state.shares}"

    def process_cash_interest(self, cash_return: float) -> float:
        """Accrues interest on spendable cash over calendar gap (Regulation §10)."""
        if self.state.cash_available <= 0.0 or cash_return <= 0.0:
            return 0.0

        interest = self.state.cash_available * cash_return
        self.state.cash_available += interest
        self.state.cumulative_cash_interest += interest
        self.state.nav += interest
        return interest

    def process_corporate_actions(
        self,
        dividend_amt: float,
        split_factor: float,
        is_pay_date: bool = True,
    ) -> Dict[str, float]:
        """Applies split and dividend adjustments before open trading (Regulation §11)."""
        events = {"split_shares_added": 0, "dividend_entitlement": 0.0, "dividend_cash_credited": 0.0}

        # 1. Stock split adjustment: adjust shares, keep economic NAV unchanged
        if split_factor > 0.0 and abs(split_factor - 1.0) > 1e-6 and self.state.shares > 0:
            new_shares = int(math.floor(self.state.shares * split_factor))
            events["split_shares_added"] = new_shares - self.state.shares
            self.state.shares = new_shares

        # 2. Ordinary cash dividend
        if dividend_amt > 0.0 and self.state.shares > 0:
            entitlement = float(self.state.shares) * float(dividend_amt)
            events["dividend_entitlement"] = entitlement
            self.state.dividend_receivable += entitlement
            self.state.cumulative_dividends += entitlement

        # 3. Payment date: convert receivable into spendable cash
        if is_pay_date and self.state.dividend_receivable > 0.0:
            pay_amt = self.state.dividend_receivable
            self.state.cash_available += pay_amt
            self.state.dividend_receivable = 0.0
            events["dividend_cash_credited"] = pay_amt

        return events

    def snapshot_pretrade_nav(self, open_price: float) -> float:
        """Computes and locks pre-trade open NAV (Regulation §8, V^{open,-})."""
        market_val = float(self.state.shares) * float(open_price)
        self.state.market_value = market_val
        self.state.nav = self.state.cash_available + market_val + self.state.dividend_receivable
        self.state.nav_open_pre = self.state.nav
        self.assert_nav_identity(open_price, "pre-trade open")
        return self.state.nav_open_pre

    def execute_target_weight(
        self,
        action_id: int,
        open_price: float,
    ) -> Dict[str, Any]:
        """Executes target weight rebalancing at open_price (Regulation §9)."""
        target_w = self.action_weights[action_id]
        v_pre = self.state.nav_open_pre
        p = float(open_price)

        # Ideal target shares
        q_star = int(math.floor((target_w * v_pre) / p))

        # No-trade rule (Regulation §9.1)
        if q_star == self.state.shares:
            delta_q = 0
            trade_notional = 0.0
            cost = 0.0
        else:
            delta_q = q_star - self.state.shares
            trade_notional = delta_q * p
            cost = self.cost_rate * abs(trade_notional)
            cash_post = self.state.cash_available - trade_notional - cost

            # Guard against negative cash due to transaction costs (Regulation §9 step 6)
            while cash_post < -1e-8 and q_star > 0 and delta_q > 0:
                q_star -= 1
                delta_q = q_star - self.state.shares
                trade_notional = delta_q * p
                cost = self.cost_rate * abs(trade_notional)
                cash_post = self.state.cash_available - trade_notional - cost

            self.state.shares = q_star
            self.state.cash_available = cash_post
            self.state.cumulative_transaction_cost += cost

        # Update post-trade NAV (Regulation §8, V^{open,+})
        market_val = float(self.state.shares) * p
        self.state.market_value = market_val
        self.state.nav = self.state.cash_available + market_val + self.state.dividend_receivable
        self.state.nav_open_post = self.state.nav

        realized_w = market_val / self.state.nav if self.state.nav > 0 else 0.0
        self.assert_nav_identity(p, "post-trade open")

        return {
            "action_id": action_id,
            "target_weight": target_w,
            "realized_weight": realized_w,
            "shares_post": self.state.shares,
            "delta_shares": delta_q,
            "trade_notional": trade_notional,
            "transaction_cost": cost,
            "cash_post_trade": self.state.cash_available,
            "nav_open_post": self.state.nav_open_post,
        }

    def snapshot_close_nav(self, close_price: float) -> float:
        """Updates and snapshots audit close NAV (Regulation §8, V^{close})."""
        market_val = float(self.state.shares) * float(close_price)
        self.state.market_value = market_val
        self.state.nav = self.state.cash_available + market_val + self.state.dividend_receivable
        self.state.nav_close = self.state.nav
        self.assert_nav_identity(close_price, "close")
        return self.state.nav_close
