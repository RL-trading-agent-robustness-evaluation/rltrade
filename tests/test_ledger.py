"""Gate L1 Deterministic Unit Tests per Regulation V1 §17."""

import math
import pytest
import numpy as np
from src.ledger.core import DoubleEntryLedger


def test_l1_cash_only():
    """Test 1: Cash-only run accrues interest without transaction costs."""
    ledger = DoubleEntryLedger(initial_cash=1_000_000.0, cost_bps=10.0)
    open_price = 300.0
    close_price = 305.0

    # Step 1: accrue 1-day interest
    interest = ledger.process_cash_interest(cash_return=0.0001)
    v_pre = ledger.snapshot_pretrade_nav(open_price)
    assert ledger.state.shares == 0
    assert abs(v_pre - (1_000_000.0 + interest)) < 1e-6

    # Action 0: 0% weight
    trade_info = ledger.execute_target_weight(action_id=0, open_price=open_price)
    assert trade_info["transaction_cost"] == 0.0
    assert trade_info["shares_post"] == 0
    assert ledger.state.shares == 0


def test_l1_buy_100_percent_and_hold():
    """Test 2 & 5: Buy 100% on day 1, no duplicate cost when repeating action 2."""
    ledger = DoubleEntryLedger(initial_cash=1_000_000.0, cost_bps=10.0)
    open_1 = 250.0

    ledger.snapshot_pretrade_nav(open_1)
    t1 = ledger.execute_target_weight(action_id=2, open_price=open_1)
    assert t1["shares_post"] > 0
    assert t1["transaction_cost"] > 0.0
    shares_held = t1["shares_post"]

    # Day 2: same action 2, same open price -> target shares equal current shares -> cost must be 0
    open_2 = 250.0
    ledger.snapshot_pretrade_nav(open_2)
    t2 = ledger.execute_target_weight(action_id=2, open_price=open_2)
    assert t2["delta_shares"] == 0
    assert t2["transaction_cost"] == 0.0
    assert ledger.state.shares == shares_held


def test_l1_sell_100_to_0():
    """Test 3: Sell 100% back to 0%."""
    ledger = DoubleEntryLedger(initial_cash=1_000_000.0, cost_bps=10.0)
    open_1 = 200.0
    ledger.snapshot_pretrade_nav(open_1)
    ledger.execute_target_weight(action_id=2, open_price=open_1)
    assert ledger.state.shares > 0

    # Sell all
    open_2 = 210.0
    ledger.snapshot_pretrade_nav(open_2)
    t2 = ledger.execute_target_weight(action_id=0, open_price=open_2)
    assert t2["shares_post"] == 0
    assert ledger.state.shares == 0
    assert ledger.state.cash_available > 1_000_000.0  # Made profit on sale minus cost


def test_l1_integer_shares_rounding():
    """Test 4: Candidate target shares must be integer floor."""
    ledger = DoubleEntryLedger(initial_cash=1_000_000.0, cost_bps=10.0)
    open_price = 333.33
    v_pre = ledger.snapshot_pretrade_nav(open_price)

    t = ledger.execute_target_weight(action_id=1, open_price=open_price)  # 50% target
    expected_shares = int(math.floor((0.5 * v_pre) / open_price))
    assert t["shares_post"] == expected_shares
    assert isinstance(ledger.state.shares, int)


def test_l1_synthetic_split_preserves_economic_nav():
    """Test 9: 2-for-1 split doubles shares without changing NAV."""
    ledger = DoubleEntryLedger(initial_cash=1_000_000.0, cost_bps=10.0)
    open_1 = 400.0
    ledger.snapshot_pretrade_nav(open_1)
    ledger.execute_target_weight(action_id=2, open_price=open_1)
    old_shares = ledger.state.shares

    # Synthetic 2-for-1 split: shares double, price halves
    events = ledger.process_corporate_actions(dividend_amt=0.0, split_factor=2.0)
    assert events["split_shares_added"] == old_shares
    assert ledger.state.shares == old_shares * 2

    # Halved open price Mark-to-Market
    v_split = ledger.snapshot_pretrade_nav(open_price=200.0)
    # Economic NAV should match
    ledger.assert_nav_identity(200.0, "post-split")


def test_l1_dividend_receivable_and_payment():
    """Test 7 & 8: Dividend receivable on ex-date, converted to cash on payment date."""
    ledger = DoubleEntryLedger(initial_cash=1_000_000.0, cost_bps=10.0)
    ledger.state.shares = 1000
    ledger.state.cash_available = 500_000.0
    open_p = 500.0

    # Ex-date with $2.00 dividend, not payment date
    events = ledger.process_corporate_actions(dividend_amt=2.0, split_factor=1.0, is_pay_date=False)
    assert events["dividend_entitlement"] == 2000.0
    assert ledger.state.dividend_receivable == 2000.0
    assert ledger.state.cash_available == 500_000.0  # Cash not spendable yet

    # Check NAV includes receivable
    v_ex = ledger.snapshot_pretrade_nav(open_p)
    assert v_ex == 500_000.0 + 1000 * 500.0 + 2000.0

    # Payment date occurs
    events_pay = ledger.process_corporate_actions(dividend_amt=0.0, split_factor=1.0, is_pay_date=True)
    assert events_pay["dividend_cash_credited"] == 2000.0
    assert ledger.state.dividend_receivable == 0.0
    assert ledger.state.cash_available == 502_000.0  # Spendable now
