#!/usr/bin/env python3
"""
Simple runner script for backtesting.
Example usage:
    python run_backtest.py --symbol BTC-USD --start 2023-01-01 --end 2024-01-01
"""
import argparse
from datetime import datetime, timedelta
from backtesting import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="Run UltraThink backtesting")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading symbol (default: BTC-USD)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (default: 1 year ago)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate (default: 0.001)")
    parser.add_argument("--output", default="backtest_report.json", help="Output report file")
    parser.add_argument("--skip-days", type=int, default=200, help="Skip first N days for indicator warmup (default: 200)")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API for agents (requires OPENAI_API_KEY)")

    args = parser.parse_args()

    # Set default dates if not provided
    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")
    if args.start is None:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        start_date = end_date - timedelta(days=365)
        args.start = start_date.strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print("UltraThink Backtesting Engine")
    print(f"{'='*70}\n")
    print(f"Symbol:          {args.symbol}")
    print(f"Period:          {args.start} to {args.end}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Commission:      {args.commission*100:.2f}%")
    print(f"OpenAI API:      {'Enabled' if args.use_openai else 'Disabled (using mock)'}")
    print(f"\n{'='*70}\n")

    # Initialize engine
    engine = BacktestEngine(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        commission_rate=args.commission,
        use_openai=args.use_openai
    )

    try:
        # Load data
        print("Loading market data...")
        engine.load_data()

        # Run backtest
        print(f"\nRunning backtest (skipping first {args.skip_days} days)...")
        engine.run(skip_first_n=args.skip_days)

        # Print results
        engine.print_report()

        # Save report
        engine.save_report(args.output)
        print(f"\nDetailed report saved to: {args.output}")

    except Exception as e:
        print(f"\nError during backtest: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        engine.cleanup()


if __name__ == "__main__":
    main()
