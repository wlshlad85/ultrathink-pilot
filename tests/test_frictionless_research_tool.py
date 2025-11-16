from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.frictionless_research_tool import (
    build_reports,
    compute_daily_returns,
    generate_synthetic_prices,
    main,
    reports_to_dataframe,
)


def test_build_reports_with_synthetic_prices():
    prices = generate_synthetic_prices(["AAA", "BBB", "CCC"], "2024-01-01", "2024-04-01")
    returns = compute_daily_returns(prices)
    reports = build_reports(returns)

    assert len(reports) == 4
    summary = reports_to_dataframe(reports)
    assert set(summary.index) == {
        "Equal_Weighted",
        "Inverse_Variance",
        "Max_Sharpe",
        "Momentum_Tilt",
    }
    # All portfolios should avoid pathological weights resulting in absurd returns.
    assert (summary["TotalReturn"] > -0.9).all()
    assert (summary["TotalReturn"] < 5).all()


def test_main_with_csv(tmp_path):
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    data = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "close": list(range(100, 100 + len(dates))) + list(range(200, 200 + len(dates))),
        }
    )
    csv_path = tmp_path / "prices.csv"
    data.to_csv(csv_path, index=False)

    summary = main(
        [
            "--tickers",
            "AAA",
            "BBB",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-15",
            "--csv",
            str(csv_path),
        ]
    )

    assert "Equal_Weighted" in summary.index
    assert (summary >= -1).all().all()
