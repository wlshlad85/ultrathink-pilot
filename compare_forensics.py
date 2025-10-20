#!/usr/bin/env python3
"""Compare old vs new model forensics results"""
import pandas as pd

# Load all decision files for new model
new_periods = {
    '2020 COVID': 'forensics_regime_aware/2020_covid_crash/decisions.csv',
    '2021 Q4': 'forensics_regime_aware/2021_q4_peak/decisions.csv',
    '2022 Bear': 'forensics_regime_aware/2022_bear_market/decisions.csv',
    '2023 Recovery': 'forensics_regime_aware/2023_recovery/decisions.csv',
    '2024 Bull': 'forensics_regime_aware/2024_bull_run/decisions.csv'
}

# Old model results from previous forensics
old_results = {
    '2020 COVID': {'decisions': 69, 'bad': 0, 'cost': 0},
    '2021 Q4': {'decisions': 40, 'bad': 8, 'cost': 7955},
    '2022 Bear': {'decisions': 272, 'bad': 26, 'cost': 44841},
    '2023 Recovery': {'decisions': 129, 'bad': 0, 'cost': 0},
    '2024 Bull': {'decisions': 130, 'bad': 2, 'cost': 2803}
}

print('='*100)
print('MODEL COMPARISON: OLD (Professional) vs NEW (Regime-Aware)')
print('='*100)
print()
print(f"{'Period':<15} {'Decisions':>10} | {'Old Bad':>10} {'Old Cost':>12} | {'New Bad':>10} {'New Cost':>12} | {'Δ Bad':>10} {'Δ Cost':>12}")
print('-'*100)

total_old_bad = 0
total_old_cost = 0
total_new_bad = 0
total_new_cost = 0
total_decisions = 0

for period_name, filepath in new_periods.items():
    df = pd.read_csv(filepath)
    bad_decisions = df[df['is_bad_decision'] == True]

    old = old_results[period_name]
    new_bad = len(bad_decisions)
    new_cost = bad_decisions['cost_of_mistake'].sum()

    total_decisions += len(df)
    total_old_bad += old['bad']
    total_old_cost += old['cost']
    total_new_bad += new_bad
    total_new_cost += new_cost

    delta_bad = new_bad - old['bad']
    delta_cost = new_cost - old['cost']

    delta_bad_str = f"{delta_bad:+d}" if delta_bad != 0 else "="
    delta_cost_str = f"${delta_cost:+,.0f}" if delta_cost != 0 else "="

    print(f"{period_name:<15} {len(df):10d} | {old['bad']:10d} ${old['cost']:>10,.0f} | {new_bad:10d} ${new_cost:>10,.0f} | {delta_bad_str:>10s} {delta_cost_str:>12s}")

print('-'*100)
delta_total_bad = total_new_bad - total_old_bad
delta_total_cost = total_new_cost - total_old_cost
print(f"{'TOTAL':<15} {total_decisions:10d} | {total_old_bad:10d} ${total_old_cost:>10,.0f} | {total_new_bad:10d} ${total_new_cost:>10,.0f} | {delta_total_bad:+10d} ${delta_total_cost:+12,.0f}")
print('='*100)
print()

# Calculate error rates
old_error_rate = (total_old_bad / total_decisions) * 100
new_error_rate = (total_new_bad / total_decisions) * 100

print(f"Old Model Error Rate: {old_error_rate:.2f}% ({total_old_bad}/{total_decisions})")
print(f"New Model Error Rate: {new_error_rate:.2f}% ({total_new_bad}/{total_decisions})")
print(f"Change: {new_error_rate - old_error_rate:+.2f} percentage points")
print()
print(f"Old Model Total Cost: ${total_old_cost:,.2f}")
print(f"New Model Total Cost: ${total_new_cost:,.2f}")
print(f"Change: ${total_new_cost - total_old_cost:+,.2f} ({((total_new_cost - total_old_cost) / total_old_cost * 100) if total_old_cost > 0 else 0:+.1f}%)")
