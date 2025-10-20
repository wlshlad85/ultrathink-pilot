#!/usr/bin/env python3
"""
Run Trade Decision Forensics

Main script to analyze model failures and generate comprehensive reports.
Focuses on critical periods where the model made poor decisions.

Usage:
    python run_forensics.py                    # Full analysis
    python run_forensics.py --period 2022      # 2022 bear market only
    python run_forensics.py --quick            # Quick test run
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trade_forensics import TradeForensics
from forensics_visualizer import ForensicsVisualizer
from backtesting.data_fetcher import DataFetcher


def analyze_critical_period(
    forensics: TradeForensics,
    visualizer: ForensicsVisualizer,
    period_name: str,
    start_date: str,
    end_date: str,
    description: str,
    output_dir: str = "forensics_output"
):
    """
    Analyze a specific critical period in detail.

    Args:
        forensics: TradeForensics engine
        visualizer: ForensicsVisualizer
        period_name: Name for this analysis period
        start_date: Start date
        end_date: End date
        description: Description of period characteristics
        output_dir: Base output directory
    """
    print("\n" + "="*80)
    print(f"ANALYZING: {period_name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Context: {description}")
    print("="*80)

    # Run analysis
    decisions = forensics.analyze_period(
        symbol="BTC-USD",
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        failure_threshold_pct=-5.0
    )

    # Generate report
    report = forensics.generate_report(decisions)
    report.print_summary()

    # Create period-specific output directory
    period_dir = Path(output_dir) / period_name.lower().replace(" ", "_")
    period_dir.mkdir(parents=True, exist_ok=True)

    # Export decisions to CSV
    csv_path = period_dir / "decisions.csv"
    forensics.export_decisions(str(csv_path), decisions)

    # Get market data for visualizations
    data_fetcher = DataFetcher("BTC-USD")
    data_fetcher.fetch(start_date, end_date)
    market_data = data_fetcher.data

    # Generate visualizations
    print(f"\nGenerating visualizations for {period_name}...")
    visualizer.create_full_report(
        decisions=decisions,
        market_data=market_data,
        output_dir=str(period_dir)
    )

    # Save report text
    report_path = period_dir / "report.txt"
    with open(report_path, 'w') as f:
        # Redirect print to file
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        report.print_summary()

        sys.stdout = old_stdout
        f.write(buffer.getvalue())

    print(f"\n✓ Complete analysis saved to {period_dir}/")

    return decisions, report


def run_full_analysis(
    model_path: str = "rl/models/professional/best_model.pth",
    output_dir: str = "forensics_output"
):
    """
    Run comprehensive forensics analysis on all critical periods.

    Args:
        model_path: Path to trained model
        output_dir: Output directory for all results
    """
    print("\n" + "="*80)
    print("TRADE DECISION FORENSICS - FULL ANALYSIS")
    print("="*80)
    print(f"\nModel: {model_path}")
    print(f"Output: {output_dir}/")
    print()

    # Check model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nAvailable models:")
        models_dir = Path("rl/models/professional")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                print(f"  - {model_file}")
        sys.exit(1)

    # Initialize forensics engine and visualizer
    print("Initializing forensics engine...")
    forensics = TradeForensics(model_path=model_path)
    visualizer = ForensicsVisualizer()

    # Define critical periods to analyze
    critical_periods = [
        {
            'name': '2020_COVID_Crash',
            'start': '2020-02-01',
            'end': '2020-05-31',
            'description': 'COVID-19 crash and rapid recovery. Extreme volatility.'
        },
        {
            'name': '2021_Q4_Peak',
            'start': '2021-10-01',
            'end': '2021-12-31',
            'description': 'BTC all-time high (~69k). Did model sell near peak?'
        },
        {
            'name': '2022_Bear_Market',
            'start': '2022-01-01',
            'end': '2022-12-31',
            'description': 'Major bear market. BTC: 47k → 15k (-68%). MODEL STRUGGLED HERE.'
        },
        {
            'name': '2023_Recovery',
            'start': '2023-01-01',
            'end': '2023-06-30',
            'description': 'Bear market recovery. BTC: 16k → 30k.'
        },
        {
            'name': '2024_Bull_Run',
            'start': '2024-01-01',
            'end': '2024-06-30',
            'description': 'New bull market. BTC: 44k → 60k+.'
        }
    ]

    # Analyze each period
    all_decisions = []
    all_reports = []

    for period in critical_periods:
        try:
            decisions, report = analyze_critical_period(
                forensics=forensics,
                visualizer=visualizer,
                period_name=period['name'],
                start_date=period['start'],
                end_date=period['end'],
                description=period['description'],
                output_dir=output_dir
            )
            all_decisions.extend(decisions)
            all_reports.append((period['name'], report))
        except Exception as e:
            print(f"\n⚠ ERROR analyzing {period['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate combined analysis
    print("\n" + "="*80)
    print("GENERATING COMBINED ANALYSIS")
    print("="*80)

    if all_decisions:
        # Combined report
        combined_report = forensics.generate_report(all_decisions)
        print("\nOVERALL PERFORMANCE ACROSS ALL PERIODS:")
        combined_report.print_summary()

        # Export combined decisions
        combined_csv = Path(output_dir) / "all_decisions_combined.csv"
        forensics.export_decisions(str(combined_csv), all_decisions)

        # Create comparison table across periods
        comparison_data = []
        for period_name, report in all_reports:
            comparison_data.append({
                'Period': period_name,
                'Total Decisions': report.total_decisions,
                'Bad Decisions': report.num_bad_decisions,
                'Error Rate (%)': f"{report.bad_decision_rate*100:.1f}",
                'Total Cost ($)': f"{report.total_cost_of_mistakes:,.0f}",
                'Buys': report.num_buys,
                'Sells': report.num_sells
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = Path(output_dir) / "period_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)

        print("\n" + "="*80)
        print("PERIOD COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✓ All results saved to: {output_dir}/")
    print(f"  - Individual period analyses in subdirectories")
    print(f"  - Combined data in all_decisions_combined.csv")
    print(f"  - Period comparison in period_comparison.csv")
    print()


def run_quick_test():
    """
    Quick test run on a small period.
    """
    print("\n" + "="*80)
    print("QUICK TEST - 2022 Q1 Analysis")
    print("="*80)

    model_path = "rl/models/professional/best_model.pth"

    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    forensics = TradeForensics(model_path=model_path)
    visualizer = ForensicsVisualizer()

    decisions, report = analyze_critical_period(
        forensics=forensics,
        visualizer=visualizer,
        period_name="Quick_Test_2022Q1",
        start_date="2022-01-01",
        end_date="2022-03-31",
        description="Quick test: Early 2022 bear market",
        output_dir="forensics_output"
    )

    print("\n✓ Quick test complete!")


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Run trade decision forensics analysis"
    )
    parser.add_argument(
        '--period',
        type=str,
        choices=['2020', '2021', '2022', '2023', '2024'],
        help='Analyze specific year only'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run (2022 Q1 only)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='rl/models/professional/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='forensics_output',
        help='Output directory'
    )

    args = parser.parse_args()

    try:
        if args.quick:
            run_quick_test()
        elif args.period:
            # Run single period
            forensics = TradeForensics(model_path=args.model)
            visualizer = ForensicsVisualizer()

            period_map = {
                '2020': ('2020-02-01', '2020-05-31', 'COVID crash'),
                '2021': ('2021-10-01', '2021-12-31', 'ATH peak'),
                '2022': ('2022-01-01', '2022-12-31', 'Bear market'),
                '2023': ('2023-01-01', '2023-06-30', 'Recovery'),
                '2024': ('2024-01-01', '2024-06-30', 'Bull run')
            }

            start, end, desc = period_map[args.period]
            analyze_critical_period(
                forensics=forensics,
                visualizer=visualizer,
                period_name=f"{args.period}_Analysis",
                start_date=start,
                end_date=end,
                description=desc,
                output_dir=args.output
            )
        else:
            # Run full analysis
            run_full_analysis(
                model_path=args.model,
                output_dir=args.output
            )

    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n⚠ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
