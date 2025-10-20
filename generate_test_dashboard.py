#!/usr/bin/env python3
"""
Generate Visual Test Dashboard for UltraThink System
Creates an interactive HTML dashboard showing test results and coverage.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def run_tests_with_json_output() -> Dict:
    """Run pytest with JSON output for detailed results."""
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short", "--json-report", "--json-report-file=test_results.json"],
        capture_output=True,
        text=True
    )

    # If pytest-json-report is not installed, use basic approach
    if result.returncode == 4 or "unrecognized arguments" in result.stderr:
        print("pytest-json-report not installed, using basic approach...")
        return run_tests_basic()

    # Load JSON results
    try:
        with open("test_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return run_tests_basic()


def run_tests_basic() -> Dict:
    """Run tests and parse output manually."""
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )

    # Parse output
    lines = result.stdout.split('\n')
    tests = []

    for line in lines:
        if '::' in line and ('PASSED' in line or 'FAILED' in line):
            parts = line.split('::')
            if len(parts) >= 2:
                test_file = parts[0].strip()
                rest = '::'.join(parts[1:])

                if 'PASSED' in rest:
                    status = 'passed'
                    test_name = rest.split('PASSED')[0].strip()
                elif 'FAILED' in rest:
                    status = 'failed'
                    test_name = rest.split('FAILED')[0].strip()
                else:
                    continue

                tests.append({
                    'nodeid': f"{test_file}::{test_name}",
                    'outcome': status,
                    'duration': 0
                })

    # Parse summary
    summary_line = [l for l in lines if 'passed' in l or 'failed' in l]
    if summary_line:
        summary = summary_line[-1]
    else:
        summary = "No summary available"

    return {
        'tests': tests,
        'summary': summary,
        'duration': 0,
        'exitcode': result.returncode
    }


def generate_html_dashboard(test_data: Dict) -> str:
    """Generate HTML dashboard from test results."""

    # Count test results
    tests = test_data.get('tests', [])
    total_tests = len(tests)
    passed = sum(1 for t in tests if t.get('outcome') == 'passed')
    failed = sum(1 for t in tests if t.get('outcome') == 'failed')

    # Group tests by file
    tests_by_file = {}
    for test in tests:
        nodeid = test.get('nodeid', '')
        parts = nodeid.split('::')
        if len(parts) >= 2:
            file_name = parts[0]
            test_name = '::'.join(parts[1:])

            if file_name not in tests_by_file:
                tests_by_file[file_name] = []

            tests_by_file[file_name].append({
                'name': test_name,
                'outcome': test.get('outcome', 'unknown'),
                'duration': test.get('duration', 0)
            })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UltraThink Test Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            color: #2d3748;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            color: #718096;
            font-size: 1.1rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
        }}

        .stat-card .label {{
            color: #718096;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-card.total .value {{ color: #4299e1; }}
        .stat-card.passed .value {{ color: #48bb78; }}
        .stat-card.failed .value {{ color: #f56565; }}
        .stat-card.success-rate .value {{ color: #667eea; }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
            transition: width 0.5s ease;
        }}

        .test-section {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .test-section h2 {{
            color: #2d3748;
            font-size: 1.5rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}

        .test-file {{
            margin-bottom: 25px;
        }}

        .test-file h3 {{
            color: #4a5568;
            font-size: 1.1rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .file-badge {{
            background: #edf2f7;
            color: #4a5568;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: normal;
        }}

        .test-list {{
            list-style: none;
        }}

        .test-item {{
            padding: 12px 15px;
            margin-bottom: 8px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s;
        }}

        .test-item:hover {{
            transform: translateX(5px);
        }}

        .test-item.passed {{
            background: #f0fff4;
            border-left: 4px solid #48bb78;
        }}

        .test-item.failed {{
            background: #fff5f5;
            border-left: 4px solid #f56565;
        }}

        .test-name {{
            flex: 1;
            color: #2d3748;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }}

        .test-status {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .status-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }}

        .status-badge.passed {{
            background: #48bb78;
            color: white;
        }}

        .status-badge.failed {{
            background: #f56565;
            color: white;
        }}

        .duration {{
            color: #718096;
            font-size: 0.85rem;
        }}

        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            padding: 20px;
        }}

        .success-icon {{
            font-size: 4rem;
            margin-bottom: 10px;
        }}

        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .stat-card, .test-section {{
            animation: slideIn 0.5s ease-out;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>UltraThink Test Dashboard</h1>
            <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card total">
                <div class="label">Total Tests</div>
                <div class="value">{total_tests}</div>
            </div>

            <div class="stat-card passed">
                <div class="label">Passed</div>
                <div class="value">{passed}</div>
            </div>

            <div class="stat-card failed">
                <div class="label">Failed</div>
                <div class="value">{failed}</div>
            </div>

            <div class="stat-card success-rate">
                <div class="label">Success Rate</div>
                <div class="value">{(passed/total_tests*100) if total_tests > 0 else 0:.1f}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {(passed/total_tests*100) if total_tests > 0 else 0}%"></div>
                </div>
            </div>
        </div>
"""

    # Add test results by file
    html += '        <div class="test-section">\n'
    html += '            <h2>Test Results by Module</h2>\n'

    for file_name, file_tests in sorted(tests_by_file.items()):
        file_passed = sum(1 for t in file_tests if t['outcome'] == 'passed')
        file_total = len(file_tests)

        html += f'            <div class="test-file">\n'
        html += f'                <h3>{file_name} <span class="file-badge">{file_passed}/{file_total} passed</span></h3>\n'
        html += '                <ul class="test-list">\n'

        for test in file_tests:
            outcome_class = test['outcome']
            duration_str = f"{test['duration']:.2f}s" if test['duration'] > 0 else ""

            html += f'                    <li class="test-item {outcome_class}">\n'
            html += f'                        <span class="test-name">{test["name"]}</span>\n'
            html += '                        <div class="test-status">\n'
            html += f'                            <span class="duration">{duration_str}</span>\n'
            html += f'                            <span class="status-badge {outcome_class}">{outcome_class}</span>\n'
            html += '                        </div>\n'
            html += '                    </li>\n'

        html += '                </ul>\n'
        html += '            </div>\n'

    html += '        </div>\n'

    # Add footer
    if failed == 0:
        html += '''
        <div class="footer">
            <div class="success-icon">✅</div>
            <h2>All Tests Passed!</h2>
            <p>UltraThink system is validated and ready.</p>
        </div>
'''
    else:
        html += f'''
        <div class="footer">
            <div class="success-icon">⚠️</div>
            <h2>{failed} Test(s) Failed</h2>
            <p>Please review and fix failing tests.</p>
        </div>
'''

    html += '''
    </div>
</body>
</html>
'''

    return html


def main():
    """Main function to generate test dashboard."""
    print("Running tests...")
    test_data = run_tests_basic()

    print("Generating HTML dashboard...")
    html = generate_html_dashboard(test_data)

    # Save dashboard
    output_file = Path("test_dashboard.html")
    output_file.write_text(html)

    print(f"Dashboard generated: {output_file.absolute()}")
    print(f"Open in browser: file://{output_file.absolute()}")


if __name__ == "__main__":
    main()
