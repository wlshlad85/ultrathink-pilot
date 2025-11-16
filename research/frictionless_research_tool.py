"""Frictionless trading research helper.

This module provides a compact workflow that mirrors the frictionless
trading backtest illustrated in the reference notebook.  It focuses on
portfolio construction and evaluation under the assumption that no
transaction costs are incurred ("frictionless" markets).

Running the module as a script will:

1. Load daily close prices for a list of tickers (either from Yahoo
   Finance or from a CSV file supplied by the user).
2. Construct a handful of classic frictionless portfolio allocations
   (equal weight, maximum Sharpe ratio, inverse variance, and a simple
   momentum tilt).
3. Evaluate each strategy with a concise performance table, including
   total return, Sharpe ratio, volatility, and an estimate of turnover.
4. Emit an equity curve comparison chart that makes it easy to visually
   compare the strategies.

Example usage from the repository root::

    python -m research.frictionless_research_tool \
        --tickers AAPL MSFT GOOG \
        --start 2023-01-01 --end 2024-01-01 \
        --out results.csv

If Yahoo Finance data cannot be downloaded (e.g. offline testing), the
script automatically falls back to a reproducible synthetic price
series.  This keeps the tool useful inside CI environments.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


try:  # matplotlib is optional – plotting is skipped if unavailable.
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - guard for environments without MPL
    plt = None
    _HAS_MATPLOTLIB = False


logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class StrategyReport:
    """Container for strategy evaluation results."""

    name: str
    total_return: float
    sharpe_ratio: float
    volatility: float
    average_turnover: float
    equity_curve: pd.Series

    def to_row(self) -> Dict[str, float]:
        return {
            "Strategy": self.name,
            "TotalReturn": self.total_return,
            "SharpeRatio": self.sharpe_ratio,
            "Volatility": self.volatility,
            "AverageTurnover": self.average_turnover,
        }


def load_price_data(
    tickers: Iterable[str],
    start: str,
    end: str,
    source_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Load (or synthesise) daily close prices for the given tickers.

    Args:
        tickers: Symbols to load.
        start: Inclusive start date in YYYY-MM-DD format.
        end: Inclusive end date in YYYY-MM-DD format.
        source_csv: Optional CSV with columns ``date``, ``ticker`` and
            ``close``.  If omitted, Yahoo Finance is used when
            available.  If the download fails, synthetic prices are
            generated instead.

    Returns:
        DataFrame indexed by date with tickers as columns.
    """

    tickers = list(dict.fromkeys(tickers))  # Remove duplicates while preserving order
    if not tickers:
        raise ValueError("At least one ticker symbol is required")

    if source_csv:
        logger.info("Loading prices from CSV: %s", source_csv)
        data = pd.read_csv(source_csv, parse_dates=["date"])
        missing_cols = {"date", "ticker", "close"} - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"CSV file is missing required columns: {sorted(missing_cols)}"
            )
        prices = (
            data[data["ticker"].isin(tickers)]
            .pivot(index="date", columns="ticker", values="close")
            .sort_index()
        )
        return prices.loc[start:end]

    try:
        import yfinance as yf

        logger.info(
            "Downloading prices from Yahoo Finance for %s (%s → %s)",
            ", ".join(tickers),
            start,
            end,
        )
        df = yf.download(tickers=tickers, start=start, end=end, progress=False)["Adj Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError("No data downloaded from Yahoo Finance")
        return df
    except Exception as exc:  # pragma: no cover - fallback path is hard to test
        logger.warning("Falling back to synthetic price series: %s", exc)
        return generate_synthetic_prices(tickers, start, end)


def generate_synthetic_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Create a reproducible geometric Brownian motion price path."""

    index = pd.date_range(start=start, end=end, freq="B")
    n_days = len(index)
    rng = np.random.default_rng(seed=42)
    prices = {}
    for i, ticker in enumerate(tickers):
        mu = 0.08 + 0.02 * np.sin(i)
        sigma = 0.2 + 0.05 * np.cos(i)
        dt = 1 / TRADING_DAYS_PER_YEAR
        returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)
        series = 100 * np.exp(np.cumsum(returns))
        prices[ticker] = series
    return pd.DataFrame(prices, index=index)


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price data."""

    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    if returns.isnull().any().any():
        returns = returns.fillna(0.0)
    return returns


def normalise_weights(raw_weights: np.ndarray) -> np.ndarray:
    raw_weights = np.clip(raw_weights, 0, None)
    total = raw_weights.sum()
    if np.isclose(total, 0):
        raise ValueError("Weights must sum to a positive value")
    return raw_weights / total


def equal_weight_strategy(n_assets: int) -> np.ndarray:
    return np.repeat(1.0 / n_assets, n_assets)


def inverse_variance_strategy(cov_matrix: np.ndarray) -> np.ndarray:
    inv_var = 1.0 / np.diag(cov_matrix)
    return normalise_weights(inv_var)


def max_sharpe_strategy(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    inv_cov = np.linalg.pinv(cov_matrix)
    ones = np.ones_like(mean_returns)
    weights = inv_cov @ mean_returns
    denominator = ones @ weights
    if np.isclose(denominator, 0):
        return equal_weight_strategy(len(mean_returns))
    return weights / denominator


def momentum_tilt_strategy(mean_returns: np.ndarray) -> np.ndarray:
    shifted = mean_returns - mean_returns.min()
    if np.allclose(shifted, 0):
        return equal_weight_strategy(len(mean_returns))
    return normalise_weights(shifted)


def evaluate_strategy(name: str, weights: np.ndarray, returns: pd.DataFrame) -> StrategyReport:
    weights = normalise_weights(weights)
    weights_series = pd.Series(weights, index=returns.columns)

    daily_returns = returns @ weights_series
    cumulative = (daily_returns + 1.0).cumprod()

    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()
    volatility = std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = 0.0 if np.isclose(std_daily, 0) else (mean_daily * TRADING_DAYS_PER_YEAR) / volatility

    turnover = 0.0  # Static allocations imply zero turnover in a frictionless setting.

    return StrategyReport(
        name=name,
        total_return=cumulative.iloc[-1] - 1.0,
        sharpe_ratio=sharpe,
        volatility=volatility,
        average_turnover=turnover,
        equity_curve=cumulative,
    )


def build_reports(returns: pd.DataFrame) -> List[StrategyReport]:
    mean_returns = returns.mean().to_numpy()
    cov_matrix = returns.cov().to_numpy()
    n_assets = returns.shape[1]

    strategies = {
        "Equal_Weighted": equal_weight_strategy(n_assets),
        "Max_Sharpe": max_sharpe_strategy(mean_returns, cov_matrix),
        "Inverse_Variance": inverse_variance_strategy(cov_matrix),
        "Momentum_Tilt": momentum_tilt_strategy(mean_returns),
    }

    reports = []
    for name, weights in strategies.items():
        report = evaluate_strategy(name, weights, returns)
        reports.append(report)
    return reports


def reports_to_dataframe(reports: Iterable[StrategyReport]) -> pd.DataFrame:
    df = pd.DataFrame([report.to_row() for report in reports]).set_index("Strategy")
    df.sort_index(inplace=True)
    return df


def plot_equity_curves(reports: Iterable[StrategyReport], output_dir: Optional[Path] = None) -> Optional[Path]:
    if not _HAS_MATPLOTLIB:
        logger.info("Matplotlib not available; skipping equity curve plot")
        return None

    plt.figure(figsize=(10, 6))
    for report in reports:
        report.equity_curve.plot(label=report.name)
    plt.title("Frictionless Strategy Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "equity_curves.png"
        plt.tight_layout()
        plt.savefig(path)
        logger.info("Saved equity curve plot to %s", path)
        return path

    plt.tight_layout()
    plt.show()
    return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple frictionless trading research tool")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOG"], help="Ticker symbols")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--csv", type=Path, help="Optional CSV source with date/ticker/close columns")
    parser.add_argument("--out", type=Path, help="Optional path to write summary CSV")
    parser.add_argument("--plot-dir", type=Path, help="Directory to save the equity curve plot")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    prices = load_price_data(args.tickers, args.start, args.end, args.csv)
    prices = prices.sort_index()
    logger.info("Loaded %d days of data for %d tickers", len(prices), prices.shape[1])

    returns = compute_daily_returns(prices)
    reports = build_reports(returns)
    summary = reports_to_dataframe(reports)

    print("\nStrategy Performance Summary (Frictionless)")
    print(summary.to_string(float_format=lambda x: f"{x:0.4f}"))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out)
        logger.info("Wrote summary table to %s", args.out)

    if args.plot_dir:
        plot_equity_curves(reports, args.plot_dir)

    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
