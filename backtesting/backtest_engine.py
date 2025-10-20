#!/usr/bin/env python3
"""
Main backtesting engine that orchestrates the entire backtesting process.
Integrates data fetcher, portfolio, agents (MR-SR, ERS), and metrics.
"""
import sys
import os
from pathlib import Path
import json
import tempfile
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.data_fetcher import DataFetcher
from backtesting.portfolio import Portfolio
from backtesting.metrics import PerformanceMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main backtesting engine.
    Coordinates data, agents, portfolio, and metrics.
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        use_openai: bool = False
    ):
        """
        Initialize backtest engine.

        Args:
            symbol: Trading symbol
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital
            commission_rate: Commission rate (default 0.1%)
            use_openai: Whether to use OpenAI API for agents
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.use_openai = use_openai

        # Components
        self.data_fetcher = DataFetcher(symbol)
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            symbol=symbol
        )

        # Results
        self.agent_decisions: List[Dict] = []
        self.market_data = None

        # Paths
        self.root_dir = Path(__file__).resolve().parents[1]
        self.agents_dir = self.root_dir / "agents"
        self.tmpdir = Path(tempfile.mkdtemp(prefix="backtest_"))

        logger.info(f"Backtest engine initialized: {symbol} from {start_date} to {end_date}")
        logger.info(f"Temp directory: {self.tmpdir}")

    def load_data(self):
        """Load and prepare historical market data."""
        logger.info("Loading market data...")
        self.market_data = self.data_fetcher.fetch(
            start_date=self.start_date,
            end_date=self.end_date,
            interval="1d"
        )
        self.data_fetcher.add_technical_indicators()
        logger.info(f"Loaded {len(self.market_data)} data points")

    def create_fixture_from_context(self, context: Dict, date: str) -> Dict:
        """
        Create a YAML-like fixture structure from market context.

        Args:
            context: Market context dictionary
            date: Current date

        Returns:
            Fixture dictionary for agent input
        """
        fixture = {
            "task": f"Analyze {self.symbol} market on {date} and recommend a strategy.",
            "input_mr_sr": {
                "date": date,
                "asset": self.symbol,
                "price": context["price"],
                "indicators": context["indicators"],
                "returns": context["returns"],
                "sentiment": context["sentiment"],
                "atr14": context["indicators"]["atr_14"],
            },
            "expected": {
                "action": "HOLD",  # Default, agent will override
                "strategy": "Momentum",
                "confidence_min": 0.5
            }
        }
        return fixture

    def call_mr_sr_agent(self, fixture: Dict, date: str) -> Dict:
        """
        Call the MR-SR agent with market context.

        Args:
            fixture: Market fixture data
            date: Current date

        Returns:
            Agent recommendation
        """
        # Write fixture to temp file
        fixture_path = self.tmpdir / f"fixture_{date}.json"
        with open(fixture_path, 'w') as f:
            json.dump(fixture, f, indent=2)

        # Prepare output path
        mr_out = self.tmpdir / f"mr_{date}.json"

        # Call MR-SR agent
        cmd = [
            sys.executable,
            str(self.agents_dir / "mr_sr.py"),
            "--fixture", str(fixture_path),
            "--asset", self.symbol,
            "--out", str(mr_out)
        ]

        # Set PYTHONPATH to include project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.root_dir)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            # Load result
            with open(mr_out, 'r') as f:
                mr_result = json.load(f)

            return mr_result

        except subprocess.CalledProcessError as e:
            logger.error(f"MR-SR agent failed: {e.stderr}")
            # Return fallback
            return {
                "action": {"recommendation": "HOLD"},
                "strategy": {"name": "Error"},
                "_error": str(e)
            }

    def call_ers_agent(self, mr_result: Dict, date: str) -> Dict:
        """
        Call the ERS agent to validate MR-SR recommendation.

        Args:
            mr_result: MR-SR agent output
            date: Current date

        Returns:
            ERS validation result
        """
        # Write MR result to temp file
        mr_path = self.tmpdir / f"mr_{date}.json"
        with open(mr_path, 'w') as f:
            json.dump(mr_result, f, indent=2)

        # Prepare output path
        ers_out = self.tmpdir / f"ers_{date}.json"

        # Call ERS agent
        cmd = [
            sys.executable,
            str(self.agents_dir / "ers.py"),
            "--in", str(mr_path),
            "--out", str(ers_out)
        ]

        # Set PYTHONPATH to include project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.root_dir)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            # Load result
            with open(ers_out, 'r') as f:
                ers_result = json.load(f)

            return ers_result

        except subprocess.CalledProcessError as e:
            logger.error(f"ERS agent failed: {e.stderr}")
            return {
                "decision": "approve",
                "reasons": ["ERS error, defaulting to approve"],
                "_error": str(e)
            }

    def run(self, skip_first_n: int = 200):
        """
        Run the backtest.

        Args:
            skip_first_n: Skip first N days to ensure indicators are calculated
        """
        if self.market_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Starting backtest...")
        logger.info(f"Skipping first {skip_first_n} days for indicator warmup")

        total_days = len(self.market_data)

        # Validate skip_first_n to ensure we have enough trading days
        min_trading_days = 10
        if skip_first_n >= total_days - min_trading_days:
            raise ValueError(
                f"skip_first_n ({skip_first_n}) is too large for dataset with {total_days} days. "
                f"Need at least {min_trading_days} trading days. "
                f"Reduce skip_first_n to at most {total_days - min_trading_days} or use a longer date range."
            )

        for i in range(skip_first_n, total_days):
            # Get market context for current day
            context = self.data_fetcher.get_market_context(i)
            date = context["date"]
            price = context["price"]

            # Create fixture from context
            fixture = self.create_fixture_from_context(context, date)

            # Call MR-SR agent
            mr_result = self.call_mr_sr_agent(fixture, date)

            # Call ERS agent
            ers_result = self.call_ers_agent(mr_result, date)

            # Extract recommendation
            action_obj = mr_result.get("action", {})
            recommendation = action_obj.get("recommendation", "HOLD")
            risk_percent = action_obj.get("risk_percent")

            # Apply ERS decision
            if ers_result.get("decision") == "veto":
                logger.info(f"[{date}] ERS vetoed {recommendation}: {ers_result.get('reasons')}")
                recommendation = "HOLD"

            # Execute trade
            trade = self.portfolio.execute_trade(
                action=recommendation,
                price=price,
                timestamp=date,
                risk_percent=risk_percent,
                reason=f"{mr_result.get('strategy', {}).get('name', 'Unknown')}"
            )

            # Record equity
            self.portfolio.record_equity(date)

            # Store decision
            decision_record = {
                "date": date,
                "price": price,
                "mr_recommendation": recommendation,
                "ers_decision": ers_result.get("decision"),
                "strategy": mr_result.get("strategy", {}).get("name"),
                "trade_executed": trade is not None,
                "portfolio_value": self.portfolio.get_total_value()
            }
            self.agent_decisions.append(decision_record)

            # Log progress
            if (i - skip_first_n) % 30 == 0:
                logger.info(
                    f"[{date}] Progress: {i-skip_first_n}/{total_days-skip_first_n}, "
                    f"Portfolio: ${self.portfolio.get_total_value():,.2f}, "
                    f"Action: {recommendation}"
                )

        logger.info("Backtest completed!")

    def generate_report(self) -> Dict:
        """
        Generate comprehensive backtest report.

        Returns:
            Dictionary with all results
        """
        # Get portfolio summary
        portfolio_stats = self.portfolio.get_summary_stats()

        # Calculate performance metrics
        equity_df = self.portfolio.get_equity_dataframe()
        trades_df = self.portfolio.get_trades_dataframe()

        metrics_calc = PerformanceMetrics(equity_df)
        performance_metrics = metrics_calc.get_all_metrics(trades_df)

        # Compile report
        report = {
            "backtest_config": {
                "symbol": self.symbol,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "initial_capital": self.initial_capital,
                "commission_rate": self.commission_rate,
            },
            "portfolio_stats": portfolio_stats,
            "performance_metrics": performance_metrics,
            "num_agent_decisions": len(self.agent_decisions),
            "equity_curve": equity_df.to_dict('records'),
            "trades": trades_df.to_dict('records'),
            "agent_decisions": self.agent_decisions
        }

        return report

    def print_report(self):
        """Print formatted backtest report."""
        report = self.generate_report()

        print("\n" + "="*70)
        print("BACKTEST REPORT")
        print("="*70)

        print("\n--- Configuration ---")
        config = report["backtest_config"]
        print(f"Symbol:              {config['symbol']}")
        print(f"Period:              {config['start_date']} to {config['end_date']}")
        print(f"Initial Capital:     ${config['initial_capital']:,.2f}")
        print(f"Commission Rate:     {config['commission_rate']*100:.2f}%")

        print("\n--- Portfolio Performance ---")
        stats = report["portfolio_stats"]
        print(f"Final Value:         ${stats['final_value']:,.2f}")
        print(f"Total P&L:           ${stats['total_pnl']:,.2f}")
        print(f"Total Return:        {stats['total_return_pct']:.2f}%")
        print(f"Cash:                ${stats['cash']:,.2f}")
        print(f"Position Value:      ${stats['position_value']:,.2f}")

        print("\n--- Trading Activity ---")
        print(f"Total Trades:        {stats['total_trades']}")
        print(f"Winning Trades:      {stats['winning_trades']}")
        print(f"Losing Trades:       {stats['losing_trades']}")
        print(f"Win Rate:            {stats['win_rate_pct']:.2f}%")
        print(f"Total Commission:    ${stats['total_commission_paid']:,.2f}")

        print("\n--- Risk-Adjusted Metrics ---")
        metrics = report["performance_metrics"]
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:        {metrics['calmar_ratio']:.2f}")
        print(f"Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"Volatility:          {metrics['volatility_pct']:.2f}%")

        if 'profit_factor' in metrics:
            pf = metrics['profit_factor']
            pf_str = f"{pf:.2f}" if pf != float('inf') else "Inf"
            print(f"Profit Factor:       {pf_str}")

        print("\n" + "="*70)

    def save_report(self, filepath: str):
        """Save report to JSON file."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {filepath}")

    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir)
            logger.info(f"Cleaned up temp directory: {self.tmpdir}")


if __name__ == "__main__":
    # Example usage
    engine = BacktestEngine(
        symbol="BTC-USD",
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_capital=100000.0
    )

    # Run backtest
    engine.load_data()
    engine.run()

    # Print results
    engine.print_report()

    # Save report
    engine.save_report("backtest_report.json")

    # Cleanup
    engine.cleanup()
