# Development Log

## [2026-09-24] - Project Initialization & Alignment Audit

### Status & Environment Inspection
- **Workspace**: `c:\03_大學\大三專題\rltrade`
- **Current Branch**: `master` (commits: `0938d83`, `6ad2aa8`, `298551f`)
- **Existing Files**:
  - `src/fetchPrice.py`: Tiingo API data fetcher for SPY daily data (2019-01-02 to present).
  - `data/spy_price.csv`: Tiingo daily OHLCV + divCash + splitFactor (1,935 rows).
  - `manifest/`: Manifest generator and validation script (`make_manifest.py`, `validate.py`, `data_manifest_SPY.json`).
- **Python Environment Verification**:
  - Installed packages on Python 3.14: `torch` (2.14.0+cpu), `stable-baselines3` (2.9.0), `gymnasium` (1.3.0), `pandas` (3.0.6), `scipy` (1.18.1), `pytest` (9.1.1), `pyyaml` (6.0.3), `matplotlib` (3.11.2), `requests` (2.34.2), `tensorboard` (2.21.0).
  - Configured `pyproject.toml` and installed `rltrade` in editable mode (`pip install -e .`).

### Strategic Decisions Locked via Alignment
1. **Rule Set & Action Space**:
   - **Regulation V1 [FROZEN]** is the single source of truth.
   - Long-only trading: no shorting, no leverage, no negative cash ($Cash \ge -10^{-8}$).
   - Action space: `Discrete(3)` -> Target weights: `{0: 0.0, 1: 0.5, 2: 1.0}`.
   - Execution: Next-open execution ($O_{t+1}^{true}$). Decision locked at close $t$, executed at open $t+1$.
   - Reward: Open-to-open pre-trade NAV log return $r_t = \log(V_{t+2}^{open,-} / V_{t+1}^{open,-})$. Transaction cost naturally absorbed in next-open cash balance.
2. **Data Roadmap**:
   - **Phase 1 Prototype**: Use available Tiingo SPY data (2019–2026) with standardized schema adapter, incorporating FRED DGS3MO yield and corporate actions (`divCash`, `splitFactor`).
   - **Phase 2 Canonical**: Seamlessly plug in CRSP 50-col master (`crsp_dsf_spy_tlt_gld_2009_2025.csv`) and FRED `DGS3MO.csv` for formal gates and publication benchmarks.
3. **Architecture Decoupling for Future Adversarial Extensions**:
   - `src/data/fred_cash.py`: Causal backward as-of join of daily FRED DGS3MO yield, computing calendar-day ACT/365 accruals.
   - `src/data/adapter.py`: Standardizes raw data, performs Gate D1 assertions (monotonic dates, positive prices, valid OHLC inequalities, non-negative volume).
   - `src/data/splits.py`: Splits canonical data into Train (`2019-01-02` to `2022-12-31`, 1,008 rows), Validation (`2023-01-01` to `2024-12-31`, 502 rows), and Test (`2025-01-01` to `2026-09-14`, 425 rows).
   - `src/features/indicator_engine.py`: Vectorized price-space indicator engine (OHLCV -> RSI, MACD, Bollinger Bands, ATR, rolling vol) with causal rolling 60-day z-scores and explicit internal state masking (`PERTURBABLE_MASK`).
   - `src/ledger/core.py`: Strict double-entry ledger tracking cash, shares, receivable, costs, interest, and NAV identities ($V^{open,-}, V^{open,+}, V^{close}$).
   - `src/env/trading_env.py`: Gymnasium-compliant environment exposing `market_true` and `market_obs` streams with next-open execution.

---

## [2026-09-24] - Acceptance Gates & Baseline Benchmarks

### 1. Gate L1 Deterministic Unit Tests (`tests/test_ledger.py`)
- **Status**: PASSED (6/6 in 0.12s)
  - Cash-only run accrues interest without transaction costs.
  - Buy 100% on Day 1, no duplicate cost when repeating action 2.
  - Sell 100% back to 0%.
  - Candidate target shares round down (floor integer shares).
  - Synthetic 2-for-1 split doubles shares without altering NAV.
  - Dividend receivable created on ex-date and credited to cash on payment date.

### 2. Gate E2 Environment Acceptance Tests (`tests/test_env.py`)
- **Status**: PASSED (3/3)
  - Gymnasium `check_env` validation: PASSED.
  - Stable-Baselines3 `check_env` validation: PASSED.
  - Observation bounds & finite values: PASSED (17-dim vector, zero NaNs/Infs).
  - Seed determinism: PASSED (identical actions and seeds yield identical NAV trajectories to machine precision).

### 3. Gate E1 Scripted Baselines Performance (`src/baselines/scripted.py`)
Tested across the full 2019–2026 SPY dataset (1,935 rows):
| Strategy | Total Return (%) | Annual Return (%) | Annual Vol (%) | Annual Sharpe | Max Drawdown (%) | Total Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Cash (0%)** | +53.43% | +5.74% | 0.36% | 15.37 | 0.00% | \$0.00 |
| **Buy & Hold (100%)** | +284.13% | +19.17% | 19.24% | 1.01 | 33.34% | \$1,437.50 |
| **Static 50/50** | +149.88% | +12.67% | 9.62% | 1.29 | 17.90% | \$6,400.50 |
| **Alternating (0/100)** | -68.04% | -13.81% | 13.37% | -1.04 | 69.18% | \$1,266,373.56 |
| **MA Crossover (20/50)**| +112.70% | +10.33% | 12.23% | 0.87 | 25.52% | \$46,062.21 |
| **Random Agent** | -38.08% | -6.05% | 12.17% | -0.45 | 50.15% | \$663,727.30 |

> **Key Realism Guard Verification**: The Random Agent produces a negative return (-6.05%) and negative Sharpe (-0.45) due to transaction cost friction, confirming that the environment does not have lookahead leakage or transaction cost omissions.

---

## [2026-09-24] - Victim V1 PPO Training & Validation Results

### Training Run Details
- **Script**: `src/agent/train_ppo.py`
- **Seed**: 42
- **Timesteps**: 25,000 steps on Train split (2019–2022)
- **Model Checkpoint**: `models/ppo_victim_v1_seed42.zip`
- **Step Ledger Output**: `experiments/ppo_runs/val_ledger_seed42.csv` (500 steps logged)

### Validation Evaluation Metrics (2023–2024 AI Bull Regime)
- **Total Return**: +32.21%
- **Annual Return**: +15.08%
- **Annual Sharpe**: 1.89
- **Max Drawdown**: 7.47%
- **Action Distribution**:
  - `Action 0 (0% Cash)`: 19.8%
  - `Action 1 (50% Half)`: 51.0%
  - `Action 2 (100% Full)`: 29.2%

### Victim Gate V1: State-Responsiveness Test
- **Normal Observation Action Distribution**: `{0: 19.8%, 1: 51.0%, 2: 29.2%}`
- **Masked Observation Action Distribution** (Market features zeroed out): `{0: 0.0%, 1: 100.0%, 2: 0.0%}`
- **Action Difference Rate**: **49.0%** (> 5% threshold)
- **Result**: **PASSED**. The policy is actively state-responsive and has not degenerated into a fixed action.

---

## [2026-09-25] - Canonical CRSP & FRED Data Ingestion & Gate D1 Audit

### 1. Checksum & Integrity Audit
The 4 canonical datasets downloaded from WRDS/FRED match Regulation V1 §2.2 SHA-256 checksums byte-for-byte:
- `crsp_dsf_spy_tlt_gld_2009_2025.csv`: SHA-256 `3448d24ac6f7ac82cb982db27187ab6daa5614f2431b824eabd670332993eb98` (12,828 rows, 50 cols, 2009-01-02 to 2025-12-31).
- `dsedist_spy_tlt_gld_2009_2025.csv`: SHA-256 `a27dbb6e56c486eeeabd1131ced492e18384695b1165f34d691766f279727969` (256 rows, 8 cols).
- `DGS3MO.csv`: SHA-256 `bfa63786acff1fab5d19b88c7f4baa281288b59028531f8567a974e1b46d4953` (4,173 rows, 2010-01-04 to 2025-12-31).
- `DTB3.csv`: SHA-256 `87a720f7c485929038db6591ae6a77a55cfe7c03097eeff5a88f73cac7740556` (4,173 rows, 2010-01-04 to 2025-12-31).

### 2. Ingestion & Quality Gates
- **Gate D1 (Data Quality)**: Verified 0 NaNs, 0 OHLC inequality violations across all 4,276 rows for each asset:
  - SPY (`PERMNO 84398`): 4,276 rows (68 dividend distributions).
  - TLT (`PERMNO 89468`): 4,276 rows (192 dividend distributions).
  - GLD (`PERMNO 90448`): 4,276 rows (0 dividend distributions).
- **Generated Canonical Datasets** in `data/interim/`:
  - `spy_canonical_crsp.csv` (4,276 rows, 2009–2025)
  - `tlt_canonical_crsp.csv` (4,276 rows, 2009–2025)
  - `gld_canonical_crsp.csv` (4,276 rows, 2009–2025)
- **Generated Regulation V1 Academic Splits** in `data/processed/crsp_spy/`:
  - Train: 2,264 rows (`2010-01-04` to `2018-12-31`)
  - Validation: 757 rows (`2019-01-01` to `2021-12-31`)
  - Locked Test: 1,003 rows (`2022-01-01` to `2025-12-31`)

### 3. Compute Infrastructure & Colab Runner
- Created `notebooks/colab_runner.ipynb` to support offloading heavy multi-seed training sweeps to Google Colab GPU / High-RAM instances.
- Pre-configured with environment checks, Google Drive mounting / Git cloning, unit test validation, multi-seed training execution, evaluation reports, and inline TensorBoard monitoring.
