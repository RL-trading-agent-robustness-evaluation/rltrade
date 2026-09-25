"""Price-space indicator engine adhering to Proposal §4.1, §4.2, §5 and Regulation V1.

Computes self-consistent technical indicators from raw OHLCV series.
In Phase 2, adversarial perturbations directly modify OHLCV in price space,
and this engine recomputes coherent indicators without look-ahead leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Explicit mask for adversarial perturbation: market features True, internal bookkeeping False (Proposal M4)
MARKET_DIMS = 15
INTERNAL_DIMS = 2
TOTAL_STATE_DIMS = MARKET_DIMS + INTERNAL_DIMS

PERTURBABLE_MASK = np.array([True] * MARKET_DIMS + [False] * INTERNAL_DIMS, dtype=bool)

FEATURE_NAMES = [
    "ret_1d",
    "hl_spread",
    "co_spread",
    "vol_ratio_20",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_pct_b",
    "bb_width",
    "atr_14_rel",
    "rolling_vol_20",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "position_weight",
    "cash_ratio",
]


def compute_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Computes Exponential Moving Average causally."""
    alpha = 2.0 / (span + 1.0)
    ema = np.zeros_like(series, dtype=np.float64)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = alpha * series[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Computes RSI(14) using Wilder's exponential smoothing."""
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n <= period:
        return rsi

    delta = np.diff(close)
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)

    # Initial SMA
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Computes Average True Range (ATR)."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < 2:
        return atr

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_cp = abs(high[i] - close[i - 1])
        l_cp = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_cp, l_cp)

    atr = compute_ema(tr, span=period)
    return atr


def compute_price_features(
    open_p: np.ndarray,
    high_p: np.ndarray,
    low_p: np.ndarray,
    close_p: np.ndarray,
    volume_p: np.ndarray,
) -> np.ndarray:
    """Vectorized causal indicator recomputation from raw price series.
    
    Returns array of shape (N, 15).
    """
    n = len(close_p)
    features = np.zeros((n, MARKET_DIMS), dtype=np.float64)

    eps = 1e-8

    # 1. 1-day log return
    ret_1d = np.zeros(n, dtype=np.float64)
    ret_1d[1:] = np.log(close_p[1:] / np.maximum(close_p[:-1], eps))
    features[:, 0] = ret_1d

    # 2. High-Low normalized spread
    features[:, 1] = (high_p - low_p) / np.maximum(close_p, eps)

    # 3. Close-Open normalized spread
    features[:, 2] = (close_p - open_p) / np.maximum(open_p, eps)

    # 4. Volume ratio to 20-day SMA
    vol_sma20 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - 19)
        vol_sma20[i] = np.mean(volume_p[start : i + 1])
    features[:, 3] = np.log(np.maximum(volume_p, 1.0) / np.maximum(vol_sma20, 1.0))

    # 5. RSI(14) centered to [-1, 1]
    rsi = compute_rsi(close_p, period=14)
    features[:, 4] = (rsi - 50.0) / 50.0

    # 6-8. MACD(12, 26, 9)
    ema12 = compute_ema(close_p, span=12)
    ema26 = compute_ema(close_p, span=26)
    macd_line = (ema12 - ema26) / np.maximum(close_p, eps)
    macd_signal = compute_ema(macd_line, span=9)
    macd_hist = macd_line - macd_signal
    features[:, 5] = macd_line
    features[:, 6] = macd_signal
    features[:, 7] = macd_hist

    # 9-10. Bollinger Bands (20, 2)
    bb_pct_b = np.zeros(n, dtype=np.float64)
    bb_width = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - 19)
        window = close_p[start : i + 1]
        mean = np.mean(window)
        std = np.std(window)
        bb_pct_b[i] = (close_p[i] - mean) / (2.0 * std + eps)
        bb_width[i] = 4.0 * std / (mean + eps)
    features[:, 8] = bb_pct_b
    features[:, 9] = bb_width

    # 11. Relative ATR(14)
    atr = compute_atr(high_p, low_p, close_p, period=14)
    features[:, 10] = atr / np.maximum(close_p, eps)

    # 12. 20-day rolling return volatility (annualized)
    rolling_vol = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - 19)
        rolling_vol[i] = np.std(ret_1d[start : i + 1]) * np.sqrt(252.0)
    features[:, 11] = rolling_vol

    # 13-15. Multi-horizon log returns (5d, 10d, 20d)
    for idx, horizon in [(12, 5), (13, 10), (14, 20)]:
        ret_h = np.zeros(n, dtype=np.float64)
        ret_h[horizon:] = np.log(close_p[horizon:] / np.maximum(close_p[:-horizon], eps))
        features[:, idx] = ret_h

    return features


def apply_causal_rolling_zscore(
    features: np.ndarray,
    window: int = 60,
    clip_val: float = 5.0,
) -> np.ndarray:
    """Applies causal rolling 60-day z-score normalization.
    
    Strictly causal: at time t, mean and std are computed only from indices max(0, t-window+1) to t.
    Zero look-ahead leakage.
    """
    n, d = features.shape
    norm_features = np.zeros_like(features, dtype=np.float32)
    eps = 1e-8

    for t in range(n):
        start = max(0, t - window + 1)
        hist = features[start : t + 1]
        mean = np.mean(hist, axis=0)
        std = np.std(hist, axis=0) + eps
        z = (features[t] - mean) / std
        norm_features[t] = np.clip(z, -clip_val, clip_val)

    return norm_features


class ObservationBuilder:
    """Builds complete 17-dimensional agent observation from market data and portfolio state."""

    def __init__(self, market_df: pd.DataFrame, norm_window: int = 60):
        self.market_df = market_df.copy().reset_index(drop=True)
        self.norm_window = norm_window

        self.open = self.market_df["open"].to_numpy(dtype=np.float64)
        self.high = self.market_df["high"].to_numpy(dtype=np.float64)
        self.low = self.market_df["low"].to_numpy(dtype=np.float64)
        self.close = self.market_df["close"].to_numpy(dtype=np.float64)
        self.volume = self.market_df["volume"].to_numpy(dtype=np.float64)

        raw_market_feats = compute_price_features(
            self.open, self.high, self.low, self.close, self.volume
        )
        self.market_features = apply_causal_rolling_zscore(raw_market_feats, window=self.norm_window)

    def get_observation(self, t: int, current_weight: float, cash_ratio: float) -> np.ndarray:
        """Returns 17-dim observation vector at close index t."""
        if t < 0 or t >= len(self.market_features):
            raise IndexError(f"Index {t} out of range [0, {len(self.market_features)-1}]")

        m_feats = self.market_features[t]
        internal_feats = np.array([current_weight, cash_ratio], dtype=np.float32)
        obs = np.concatenate([m_feats, internal_feats])
        return obs.astype(np.float32)
