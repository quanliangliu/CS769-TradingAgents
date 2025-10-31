"""
Backtesting engine for TradingAgents and baseline strategies.
"""

import pandas as pd
from typing import Dict, List
from pathlib import Path
import json


STD_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


class TradingAgentsBacktester:
    """Backtest engine for TradingAgents framework."""

    def __init__(self, trading_agents_graph, initial_capital=100000):
        self.graph = trading_agents_graph
        self.initial_capital = float(initial_capital)
        self.name = "TradingAgents"

    def backtest(self, ticker: str, start_date: str, end_date: str, data: pd.DataFrame) -> pd.DataFrame:
        # Restrict to window
        df = data.loc[start_date:end_date].copy()
        portfolio = pd.DataFrame(index=df.index)
        portfolio["close"] = df["Close"]
        if "Volume" in df.columns:
            portfolio["Volume"] = df["Volume"]

        portfolio["signal"] = 0
        portfolio["position"] = 0.0
        portfolio["cash"] = self.initial_capital
        portfolio["shares"] = 0.0
        portfolio["portfolio_value"] = self.initial_capital

        decisions: List[Dict] = []

        print(f"\nRunning TradingAgents backtest on {ticker} from {start_date} to {end_date}")
        print(f"Total trading days: {len(df)}")
        print("-" * 80)

        for i, (date, row) in enumerate(df.iterrows()):
            date_str = date.strftime("%Y-%m-%d")
            price = float(row["Close"])

            # Get decision
            try:
                print(f"\n[{i+1}/{len(df)}] {date_str} ... ", end="")
                final_state, decision = self.graph.propagate(ticker, date_str)
                print(f"Decision: {decision}")
                signal = self._parse_decision(decision)
                decisions.append({"date": date_str, "decision": decision, "signal": signal, "price": price})

            except Exception as e:
                print(f"Error: {e}")
                signal = 0
                decisions.append({"date": date_str, "decision": "ERROR", "signal": 0, "price": price, "error": str(e)})

            # Previous day state
            if i > 0:
                prev_cash = float(portfolio["cash"].iloc[i - 1])
                prev_shares = float(portfolio["shares"].iloc[i - 1])
                prev_pos = float(portfolio["position"].iloc[i - 1])
            else:
                prev_cash = self.initial_capital
                prev_shares = 0.0
                prev_pos = 0.0

            cash, shares, position = prev_cash, prev_shares, prev_pos

            # Execute: BUY opens/keeps long with all cash; SELL closes to cash; HOLD keeps.
            if signal == 1 and prev_pos <= 0:
                # Go long full notional
                shares = cash / price if price > 0 else 0.0
                cash = 0.0
                position = 1.0
            elif signal == -1 and prev_pos > 0:
                # Exit long to cash (no shorting here; paper's figs show short arrows,
                # but transactions table is still long/flat in our public code)
                cash = shares * price
                shares = 0.0
                position = 0.0
            else:
                # Hold current stance
                position = prev_pos

            portval = cash + shares * price

            portfolio.loc[date, "signal"] = signal
            portfolio.loc[date, "position"] = position
            portfolio.loc[date, "cash"] = cash
            portfolio.loc[date, "shares"] = shares
            portfolio.loc[date, "portfolio_value"] = portval

        # Returns
        portfolio["market_return"] = portfolio["close"].pct_change().fillna(0.0)
        portfolio["portfolio_return"] = portfolio["portfolio_value"].pct_change().fillna(0.0)
        portfolio["strategy_return"] = portfolio["portfolio_return"]
        portfolio["cumulative_return"] = (1.0 + portfolio["strategy_return"]).cumprod()

        self._save_decisions_log(ticker, decisions, start_date, end_date)
        return portfolio

    def _parse_decision(self, decision: str) -> int:
        """
        Parse decision to signal.
        We interpret:
          - contains 'BUY' or 'LONG' -> 1
          - contains 'SELL' or 'EXIT' -> -1  (we use -1 as 'close to cash' here)
          - otherwise HOLD -> 0
        """
        d = str(decision).upper()
        if "BUY" in d or "LONG" in d:
            return 1
        if "SELL" in d or "EXIT" in d or "CLOSE" in d:
            return -1
        return 0

    def _save_decisions_log(self, ticker: str, decisions: List[Dict], start_date: str, end_date: str):
        out = Path(f"eval_results/{ticker}/TradingAgents_backtest")
        out.mkdir(parents=True, exist_ok=True)
        fp = out / f"decisions_{start_date}_to_{end_date}.json"
        with open(fp, "w") as f:
            json.dump(decisions, f, indent=2)
        print(f"\nDecisions log saved to: {fp}")


class BacktestEngine:
    """Engine to run and compare multiple strategies."""

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.data = data
        self.initial_capital = float(initial_capital)
        self.results: Dict[str, pd.DataFrame] = {}

    def run_strategy(self, strategy, start_date: str = None, end_date: str = None, label = None) -> pd.DataFrame:
        data_filtered = self.data.loc[start_date:end_date] if (start_date and end_date) else self.data
        print(f"\nRunning {strategy.name}...")
        portfolio = strategy.backtest(data_filtered)
        self.results[label or strategy.name] = portfolio
        return portfolio

    def run_all_strategies(self, strategies: Dict, start_date: str = None, end_date: str = None):
        for name, strategy in strategies.items():
            try:
                self.run_strategy(strategy, start_date, end_date)
                print(f"✓ {name} completed")
            except Exception as e:
                print(f"✗ {name} failed: {e}")

    def get_results(self) -> Dict[str, pd.DataFrame]:
        return self.results


def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        # Normalize accidental ('A','A','P','L') / ['A','A','P','L']
        if isinstance(ticker, (list, tuple)) and all(isinstance(c, str) and len(c) == 1 for c in ticker):
            ticker = "".join(ticker)

        if not isinstance(ticker, str):
            raise ValueError("Pass a single ticker symbol as a string, e.g., 'AAPL'.")

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def standardize_single_ticker(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Return a single-ticker OHLCV DataFrame with simple columns.
       Works with yfinance single or multi-ticker outputs.
    """
    df = df.copy()

    # If columns are MultiIndex (common with multi-ticker yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # Figure out which level is the field (Open/High/...) and which is ticker
        lvl0 = set(map(str, df.columns.get_level_values(0)))
        lvl1 = set(map(str, df.columns.get_level_values(1)))
        if len(STD_FIELDS & lvl0) > 0:
            field_level, ticker_level = 0, 1
        elif len(STD_FIELDS & lvl1) > 0:
            field_level, ticker_level = 1, 0
        else:
            raise ValueError("Cannot detect OHLCV field level in MultiIndex columns.")

        available = list(pd.Index(df.columns.get_level_values(ticker_level)).unique())

        # Normalize weird ticker inputs like ('A','A','P','L') -> 'AAPL'
        if isinstance(ticker, (list, tuple)) and all(isinstance(c, str) and len(c) == 1 for c in ticker):
            ticker = "".join(ticker)
        if ticker is None:
            if len(available) != 1:
                raise ValueError(f"Multi-ticker DataFrame. Pick one with ticker=..., available={available}")
            ticker = available[0]
        if str(ticker) not in map(str, available):
            raise ValueError(f"Ticker {ticker!r} not in columns. Available: {available}")

        # Slice to that ticker and drop the ticker level
        df = df.xs(ticker, axis=1, level=ticker_level)

    # Map Adj Close -> Close if Close missing
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    # Final sanity
    req = ["Open", "High", "Low", "Close"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Data missing columns: {missing}")

    # Ensure 'Close' is a Series (not 1-col DataFrame)
    close = df["Close"]
    if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
        df["Close"] = close.iloc[:, 0]

    return df