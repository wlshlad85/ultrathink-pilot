"""
Load testing script for Inference API.
Tests P95 latency under realistic load.
"""
import asyncio
import aiohttp
import time
import numpy as np
from typing import List, Dict
import argparse
import json
from datetime import datetime


class LoadTester:
    """Load testing utility."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.latencies: List[float] = []
        self.errors: List[Dict] = []
        self.responses: List[Dict] = []

    async def make_request(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        """Make a single prediction request."""
        start_time = time.time()

        try:
            async with session.post(
                f"{self.base_url}/api/v1/predict",
                json={
                    "symbol": symbol,
                    "risk_check": True,
                    "explain": False
                },
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                latency = (time.time() - start_time) * 1000  # ms
                data = await response.json()

                result = {
                    'status': response.status,
                    'latency_ms': latency,
                    'success': response.status == 200,
                    'symbol': symbol,
                    'timestamp': datetime.utcnow().isoformat()
                }

                if response.status == 200:
                    result['action'] = data.get('action')
                    result['confidence'] = data.get('confidence')
                    result['server_latency_ms'] = data.get('metadata', {}).get('latency_ms')

                return result

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                'status': 0,
                'latency_ms': latency,
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat()
            }

    async def run_batch(
        self,
        n_requests: int,
        concurrent: int = 10,
        symbols: List[str] = None
    ):
        """
        Run a batch of requests.

        Args:
            n_requests: Total number of requests
            concurrent: Number of concurrent requests
            symbols: List of symbols to test (rotates through)
        """
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        print(f"\n{'='*60}")
        print(f"Load Test Configuration")
        print(f"{'='*60}")
        print(f"Total requests:      {n_requests}")
        print(f"Concurrent requests: {concurrent}")
        print(f"Target endpoint:     {self.base_url}/api/v1/predict")
        print(f"Symbols:             {', '.join(symbols)}")
        print(f"{'='*60}\n")

        async with aiohttp.ClientSession() as session:
            # Create request tasks
            tasks = []
            for i in range(n_requests):
                symbol = symbols[i % len(symbols)]
                tasks.append(self.make_request(session, symbol))

            # Execute with concurrency limit
            print(f"Starting load test...")
            start_time = time.time()

            # Run in batches
            results = []
            for i in range(0, len(tasks), concurrent):
                batch = tasks[i:i + concurrent]
                batch_results = await asyncio.gather(*batch)
                results.extend(batch_results)

                # Progress indicator
                progress = (i + len(batch)) / n_requests * 100
                print(f"  Progress: {progress:.1f}% ({i + len(batch)}/{n_requests})", end='\r')

            total_time = time.time() - start_time
            print(f"\n\nCompleted in {total_time:.2f}s")

            # Store results
            for result in results:
                if result['success']:
                    self.latencies.append(result['latency_ms'])
                    self.responses.append(result)
                else:
                    self.errors.append(result)

    def print_results(self):
        """Print load test results."""
        print(f"\n{'='*60}")
        print(f"Load Test Results")
        print(f"{'='*60}")

        total_requests = len(self.latencies) + len(self.errors)
        success_rate = len(self.latencies) / total_requests * 100 if total_requests > 0 else 0

        print(f"\nRequest Statistics:")
        print(f"  Total requests:  {total_requests}")
        print(f"  Successful:      {len(self.latencies)} ({success_rate:.1f}%)")
        print(f"  Failed:          {len(self.errors)}")

        if self.latencies:
            latencies = np.array(self.latencies)
            mean = np.mean(latencies)
            median = np.median(latencies)
            std = np.std(latencies)
            p50 = np.percentile(latencies, 50)
            p75 = np.percentile(latencies, 75)
            p90 = np.percentile(latencies, 90)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            min_lat = np.min(latencies)
            max_lat = np.max(latencies)

            print(f"\nLatency Statistics (ms):")
            print(f"  Mean:    {mean:8.2f}")
            print(f"  Median:  {median:8.2f}")
            print(f"  Std Dev: {std:8.2f}")
            print(f"  Min:     {min_lat:8.2f}")
            print(f"  Max:     {max_lat:8.2f}")
            print(f"\nPercentiles (ms):")
            print(f"  P50:     {p50:8.2f}")
            print(f"  P75:     {p75:8.2f}")
            print(f"  P90:     {p90:8.2f}")
            print(f"  P95:     {p95:8.2f}  {'✓' if p95 < 50 else '⚠'} (target: <50ms)")
            print(f"  P99:     {p99:8.2f}")

            # Performance evaluation
            print(f"\nPerformance Evaluation:")
            if p95 < 50:
                print(f"  ✓ PASS: P95 latency {p95:.2f}ms < 50ms target")
            elif p95 < 100:
                print(f"  ⚠ ACCEPTABLE: P95 latency {p95:.2f}ms < 100ms (relaxed target)")
            else:
                print(f"  ✗ FAIL: P95 latency {p95:.2f}ms exceeds 100ms")

        if self.errors:
            print(f"\nErrors:")
            error_types = {}
            for error in self.errors:
                error_msg = error.get('error', 'Unknown')
                error_types[error_msg] = error_types.get(error_msg, 0) + 1

            for error_type, count in error_types.items():
                print(f"  {error_type}: {count}")

        # Action distribution
        if self.responses:
            actions = [r['action'] for r in self.responses if 'action' in r]
            if actions:
                print(f"\nAction Distribution:")
                for action in ['BUY', 'SELL', 'HOLD']:
                    count = actions.count(action)
                    pct = count / len(actions) * 100
                    print(f"  {action:5s}: {count:4d} ({pct:5.1f}%)")

        print(f"\n{'='*60}\n")

    def save_results(self, filename: str):
        """Save results to JSON file."""
        results = {
            'summary': {
                'total_requests': len(self.latencies) + len(self.errors),
                'successful': len(self.latencies),
                'failed': len(self.errors),
                'success_rate': len(self.latencies) / (len(self.latencies) + len(self.errors)) if (len(self.latencies) + len(self.errors)) > 0 else 0
            },
            'latency_stats': {
                'mean': float(np.mean(self.latencies)) if self.latencies else None,
                'median': float(np.median(self.latencies)) if self.latencies else None,
                'p50': float(np.percentile(self.latencies, 50)) if self.latencies else None,
                'p75': float(np.percentile(self.latencies, 75)) if self.latencies else None,
                'p90': float(np.percentile(self.latencies, 90)) if self.latencies else None,
                'p95': float(np.percentile(self.latencies, 95)) if self.latencies else None,
                'p99': float(np.percentile(self.latencies, 99)) if self.latencies else None,
            },
            'responses': self.responses,
            'errors': self.errors
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filename}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Load test Inference API')
    parser.add_argument('--url', default='http://localhost:8080',
                        help='Base URL of API')
    parser.add_argument('--requests', type=int, default=1000,
                        help='Number of requests')
    parser.add_argument('--concurrent', type=int, default=10,
                        help='Concurrent requests')
    parser.add_argument('--output', default='load_test_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    tester = LoadTester(base_url=args.url)
    await tester.run_batch(
        n_requests=args.requests,
        concurrent=args.concurrent
    )

    tester.print_results()
    tester.save_results(args.output)


if __name__ == "__main__":
    asyncio.run(main())
