"""
Baseline trading strategies for comparison.
Implements: Buy&Hold, MACD, KDJ+RSI, ZMR, SMA
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, initial_capital=100000):
        self.initial_capital = float(initial_capital)
        self.name = self.__class__.__name__

    def _close_series(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 1:
                close = close.iloc[:, 0]
            else:
                raise ValueError("Multiple 'Close' columns detected. Pass single-ticker data.")
        return pd.to_numeric(close, errors="coerce")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate *target* position by date (1 long, -1 short, 0 flat)."""
        pass

    def _prep_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        req = ["Open", "High", "Low", "Close"]
        for col in req:
            if col not in data.columns:
                raise ValueError(f"Data missing column '{col}'")
        return data.copy()

    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self._prep_ohlcv(data)
        signals = self.generate_signals(df).astype(float)
        signals = signals.clip(lower=-1, upper=1).reindex(df.index).fillna(0)

        # ONE place for hold semantics (Option A: 0 = no new signal)
        position = signals.replace(0, np.nan).ffill().fillna(0)

        close = self._close_series(df)
        market_ret = close.pct_change().fillna(0.0)
        exposure = position.shift(1).fillna(0.0)
        strat_ret = (exposure * market_ret).astype(float)

        cumret = (1.0 + strat_ret).cumprod()
        portval = self.initial_capital * cumret

        portfolio = pd.DataFrame(index=df.index)
        portfolio["signal"] = signals
        portfolio["position"] = position
        portfolio["close"] = close
        if "Volume" in df.columns:
            vol = df["Volume"]
            if isinstance(vol, pd.DataFrame) and vol.shape[1] == 1:
                vol = vol.iloc[:, 0]
            if isinstance(vol, pd.Series):
                portfolio["Volume"] = vol
        portfolio["market_return"] = market_ret
        portfolio["strategy_return"] = strat_ret
        portfolio["cumulative_return"] = cumret
        portfolio["portfolio_value"] = portval
        portfolio["trade"] = portfolio["position"].diff().fillna(0.0)
        return portfolio


class BuyAndHoldStrategy(BaseStrategy):
    """Buy at start and hold long the whole period (no short)."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        s = pd.Series(1.0, index=data.index)
        return s


class MACDStrategy(BaseStrategy):
    """
    MACD Strategy.
    Long when MACD > signal, Short when MACD < signal.
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        if "macd" not in df.columns or "macds" not in df.columns:
            df = self._calculate_macd(df)
        macd_diff = (df["macd"] - df["macds"]).fillna(0.0)
        sig = pd.Series(0.0, index=df.index)
        sig[macd_diff > 0] = 1.0
        sig[macd_diff < 0] = -1.0
        return sig

    def _calculate_macd(self, data: pd.DataFrame, fast=12, slow=26, signal=9):
        exp1 = data["Close"].ewm(span=fast, adjust=False).mean()
        exp2 = data["Close"].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macds = macd.ewm(span=signal, adjust=False).mean()
        data["macd"] = macd
        data["macds"] = macds
        data["macdh"] = macd - macds
        return data


class KDJRSIStrategy(BaseStrategy):
    """
    KDJ & RSI Strategy (classic oversold/overbought gating).
    Long when RSI<30 & K<20; Short when RSI>70 & K>80.
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        if "rsi" not in df.columns:
            df = self._calculate_rsi(df)
        if "kdj_k" not in df.columns:
            df = self._calculate_kdj(df)

        sig = pd.Series(0.0, index=df.index)
        sig[(df["rsi"] < 30) & (df["kdj_k"] < 20)] = 1.0
        sig[(df["rsi"] > 70) & (df["kdj_k"] > 80)] = -1.0
        return sig

    def _calculate_rsi(self, data: pd.DataFrame, period=14):
        # Wilder's smoothing approximation via EMA improves stability
        delta = data["Close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/period, adjust=False).mean()
        roll_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        data["rsi"] = 100 - (100 / (1 + rs))
        return data

    def _calculate_kdj(self, data: pd.DataFrame, period=9):
        low_min = data["Low"].rolling(window=period, min_periods=period).min()
        high_max = data["High"].rolling(window=period, min_periods=period).max()
        den = (high_max - low_min).replace(0, np.nan)
        rsv = 100 * (data["Close"] - low_min) / den
        k = rsv.ewm(com=2, adjust=False, min_periods=1).mean()
        d = k.ewm(com=2, adjust=False, min_periods=1).mean()
        j = 3 * k - 2 * d
        data["kdj_k"], data["kdj_d"], data["kdj_j"] = k, d, j
        return data


class ZMRStrategy(BaseStrategy):
    """
    Zero-mean reversion on z-score of Close vs rolling mean.
    """

    def __init__(self, initial_capital=100000, lookback=20, threshold=1.0):
        super().__init__(initial_capital)
        self.lookback = int(lookback)
        self.threshold = float(threshold)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._close_series(data)
        rm = close.rolling(window=self.lookback, min_periods=self.lookback).mean()
        rs = close.rolling(window=self.lookback, min_periods=self.lookback).std()
        z = (close - rm) / rs.replace(0, pd.NA)
        sig = pd.Series(0.0, index=data.index)
        sig[z < -self.threshold] = 1.0
        sig[z >  self.threshold] = -1.0
        return sig


class SMAStrategy(BaseStrategy):
    """
    SMA crossover (50/200 by default).
    """

    def __init__(self, initial_capital=100000, short_window=50, long_window=200):
        super().__init__(initial_capital)
        self.short_window = int(short_window)
        self.long_window = int(long_window)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._close_series(data)
        short = close.rolling(window=self.short_window, min_periods=self.short_window).mean()
        long_ = close.rolling(window=self.long_window, min_periods=self.long_window).mean()
        sig = pd.Series(0.0, index=data.index)
        sig[short > long_] = 1.0
        sig[short < long_] = -1.0
        return sig


def get_all_baseline_strategies(initial_capital=100000):
    """Get all baseline strategies for comparison."""
    return {
        "BuyAndHold": BuyAndHoldStrategy(initial_capital),
        "MACD": MACDStrategy(initial_capital),
        "KDJ&RSI": KDJRSIStrategy(initial_capital),
        "ZMR": ZMRStrategy(initial_capital),
        "SMA": SMAStrategy(initial_capital),
    }