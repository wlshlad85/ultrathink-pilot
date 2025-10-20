# Playwright MCP Integration for UltraThink

## Overview

UltraThink now supports **browser automation** via Playwright MCP (Model Context Protocol), enabling real-time web scraping, market monitoring, and automated trading interface testing.

### What You Can Do

- 📰 **Scrape crypto news** from CoinDesk, CoinTelegraph, etc.
- 💹 **Monitor real-time prices** across multiple exchanges
- 📊 **Capture trading dashboards** for visual analysis
- 🐦 **Analyze social media sentiment** (Twitter, Reddit)
- 🧪 **Test trading interfaces** automatically
- 🔍 **Enrich backtests** with web-sourced data

---

## Quick Start

### 1. Playwright MCP is Already Available

Claude Code includes Playwright MCP by default! No additional installation required.

### 2. Use the Python Interface

```python
from tools.playwright_mcp import PlaywrightMarketScraper

# Create scraper
scraper = PlaywrightMarketScraper()

# Get latest crypto news
news = scraper.get_crypto_news("BTC", limit=10)
for item in news:
    print(f"{item.title} - {item.sentiment}")

# Compare prices across exchanges
prices = scraper.get_exchange_prices("BTC-USD")
for exchange, data in prices.items():
    print(f"{exchange}: ${data.price:,.2f}")

# Monitor Twitter sentiment
sentiment = scraper.monitor_twitter_sentiment("BTC")
print(f"Sentiment: {sentiment['sentiment_score']:.2f}")
```

### 3. Enrich Your Backtests

```python
from backtesting.data_fetcher import DataFetcher
from tools.playwright_mcp import PlaywrightDataEnricher

# Standard backtest data
fetcher = DataFetcher("BTC-USD", "2024-01-01", "2024-12-01")
context = fetcher.get_market_context(0)

# Enrich with web data
enricher = PlaywrightDataEnricher()
enriched_context = enricher.enrich_market_context(context, "BTC")

# Now includes:
# - enriched_context['recent_news']
# - enriched_context['exchange_prices']
# - enriched_context['social_sentiment']
```

---

## Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                  UltraThink System                       │
│                                                          │
│  ┌──────────────────┐         ┌──────────────────┐     │
│  │ Python Interface │         │ Claude Code MCP   │     │
│  │ (playwright_mcp) │────────▶│ Playwright Tools  │     │
│  └──────────────────┘         └──────────────────┘     │
│           │                            │                 │
│           │                            ▼                 │
│           │                   ┌──────────────────┐      │
│           │                   │  Browser Engine  │      │
│           │                   │  (Chromium/FF)   │      │
│           │                   └──────────────────┘      │
│           │                            │                 │
│           ▼                            ▼                 │
│  ┌──────────────────┐         ┌──────────────────┐     │
│  │  DataFetcher     │         │  Web Scraping    │     │
│  │  (yfinance)      │◀────────│  (live data)     │     │
│  └──────────────────┘         └──────────────────┘     │
│           │                                              │
│           ▼                                              │
│  ┌──────────────────┐                                   │
│  │ Backtest Engine  │                                   │
│  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

### Components

1. **PlaywrightMarketScraper** (`tools/playwright_mcp.py`)
   - High-level Python interface for market data scraping
   - Wraps Claude Code's Playwright MCP tools
   - Provides typed data structures (MarketNewsItem, PriceData)

2. **PlaywrightDataEnricher** (`tools/playwright_mcp.py`)
   - Integrates web-scraped data into backtesting pipeline
   - Enriches market context with news, sentiment, prices
   - Compatible with existing DataFetcher output

3. **Claude Code Playwright MCP**
   - Built-in browser automation tools
   - Available via `mcp__playwright__*` functions
   - No manual setup required

---

## Available MCP Tools

### Navigation

```python
# mcp__playwright__playwright_navigate
# Navigate to a URL
result = navigate(url="https://coinmarketcap.com/currencies/bitcoin/")
```

### Scraping

```python
# mcp__playwright__playwright_get_visible_text
# Extract text content from page
text = get_visible_text()

# mcp__playwright__playwright_get_visible_html
# Get HTML structure
html = get_visible_html(selector=".price-section")
```

### Screenshots

```python
# mcp__playwright__playwright_screenshot
# Capture visual snapshot
screenshot = screenshot(
    name="btc_chart",
    full_page=True,
    save_png=True
)
```

### Interaction

```python
# mcp__playwright__playwright_click
# Click elements
click(selector="button.trade-now")

# mcp__playwright__playwright_fill
# Fill forms
fill(selector="input[name='amount']", value="0.1")
```

---

## Use Cases

### 1. News-Driven Trading

Scrape breaking news and adjust strategy:

```python
from tools.playwright_mcp import PlaywrightMarketScraper

scraper = PlaywrightMarketScraper()

# Get latest news
news = scraper.get_crypto_news("BTC", limit=20)

# Analyze sentiment
bullish_count = sum(1 for n in news if n.sentiment == "BULLISH")
bearish_count = sum(1 for n in news if n.sentiment == "BEARISH")

if bullish_count > bearish_count * 2:
    print("Strong bullish news sentiment - consider BUY")
elif bearish_count > bullish_count * 2:
    print("Strong bearish news sentiment - consider SELL")
else:
    print("Mixed sentiment - HOLD or analyze further")
```

### 2. Arbitrage Detection

Compare prices across exchanges:

```python
exchanges = ["Binance", "Coinbase", "Kraken", "Gemini"]
prices = scraper.get_exchange_prices("BTC-USD", exchanges)

# Find arbitrage opportunities
prices_list = [(ex, data.price) for ex, data in prices.items()]
prices_list.sort(key=lambda x: x[1])

lowest = prices_list[0]
highest = prices_list[-1]

spread_pct = ((highest[1] - lowest[1]) / lowest[1]) * 100

if spread_pct > 0.5:  # 0.5% spread
    print(f"Arbitrage opportunity!")
    print(f"Buy on {lowest[0]}: ${lowest[1]:,.2f}")
    print(f"Sell on {highest[0]}: ${highest[1]:,.2f}")
    print(f"Potential profit: {spread_pct:.2f}%")
```

### 3. Social Sentiment Analysis

Incorporate Twitter/Reddit sentiment:

```python
# Monitor multiple symbols
symbols = ["BTC", "ETH", "SOL"]

for symbol in symbols:
    sentiment = scraper.monitor_twitter_sentiment(symbol)

    print(f"\n{symbol} Social Sentiment:")
    print(f"  Score: {sentiment['sentiment_score']:.2f}/1.00")
    print(f"  Mentions: {sentiment['total_mentions']}")
    print(f"  Top keywords: {', '.join(sentiment['top_keywords'])}")

    # Trading signal based on sentiment
    if sentiment['sentiment_score'] > 0.7:
        print(f"  ⚡ Signal: STRONG BUY")
    elif sentiment['sentiment_score'] < 0.3:
        print(f"  ⚡ Signal: STRONG SELL")
    else:
        print(f"  ⚡ Signal: HOLD")
```

### 4. Enhanced Backtesting

Enrich historical backtests with news context:

```python
from backtesting.backtest_engine import BacktestEngine
from tools.playwright_mcp import PlaywrightDataEnricher

# Standard backtest
engine = BacktestEngine(
    symbol="BTC-USD",
    start_date="2024-01-01",
    end_date="2024-06-01",
    initial_capital=100000
)

# Enrich with web data
enricher = PlaywrightDataEnricher()

# Custom backtesting loop with enrichment
for i, row in engine.data.iterrows():
    if i < engine.skip_days:
        continue

    # Get standard context
    context = engine.data_fetcher.get_market_context(i)

    # Enrich with live web data
    enriched = enricher.enrich_market_context(context, "BTC")

    # Pass enriched context to agents
    # (agents can now consider news, sentiment, exchange prices)
    mr_result = engine.call_mr_sr_agent(enriched, row['Date'])
    ers_result = engine.call_ers_agent(mr_result, row['Date'])

    # Execute trade based on enriched data
    # ...
```

### 5. Automated Dashboard Monitoring

Capture and analyze trading dashboards:

```python
# Capture TradingView chart
dashboard_url = "https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDT"
screenshot_path = scraper.capture_trading_dashboard(
    url=dashboard_url,
    output_path="btc_chart.png",
    full_page=True
)

print(f"Dashboard captured: {screenshot_path}")

# Could then use image analysis (not included) to:
# - Detect chart patterns
# - Read indicator values
# - Identify support/resistance levels
```

### 6. Exchange UI Testing

Test trading interfaces for reliability:

```python
test_actions = [
    "navigate_to_spot_trading",
    "select_btc_usd_pair",
    "check_order_book_loads",
    "verify_price_updates",
    "test_limit_order_form",
    "test_market_order_form"
]

results = scraper.test_trading_interface(
    exchange_url="https://exchange.example.com",
    test_actions=test_actions
)

print(f"UI Tests: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")
print(f"Duration: {results['duration_seconds']:.1f}s")
```

---

## Integration with Existing Code

### Modify MR-SR Agent to Use Web Data

Edit `agents/mr_sr.py` to accept enriched context:

```python
# In agents/mr_sr.py
def generate_recommendation(context: Dict[str, Any]) -> Dict[str, Any]:
    # Original indicators
    rsi = context.get('rsi', 50)
    macd = context.get('macd_signal', 'NEUTRAL')

    # NEW: Web-scraped data
    news_sentiment = context.get('social_sentiment', {}).get('score', 0.5)
    recent_news = context.get('recent_news', [])

    # Adjust strategy based on news
    if news_sentiment > 0.7 and rsi < 70:
        action = "BUY"
        reasoning = f"Bullish sentiment ({news_sentiment:.2f}) + RSI not overbought"
    elif news_sentiment < 0.3 and rsi > 30:
        action = "SELL"
        reasoning = f"Bearish sentiment ({news_sentiment:.2f}) + RSI not oversold"
    else:
        action = "HOLD"
        reasoning = "Mixed signals from technical + sentiment analysis"

    return {
        "action": action,
        "reasoning": reasoning,
        "confidence": abs(news_sentiment - 0.5) * 2,  # 0-1 scale
        # ... rest of recommendation
    }
```

### Add Playwright Dependency to Tools Registry

Edit `policy/tools.yaml`:

```yaml
tools:
  # ... existing tools ...

  playwright_scraper:
    class: A  # Read-only scraping
    description: "Scrape crypto news, prices, sentiment from web sources"
    timeout: 30
    allowed_domains:
      - coindesk.com
      - cointelegraph.com
      - coinmarketcap.com
      - binance.com
      - twitter.com

  playwright_screenshot:
    class: A  # Read-only capture
    description: "Capture trading dashboard screenshots for analysis"
    timeout: 15
```

---

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Playwright MCP Configuration
PLAYWRIGHT_HEADLESS=true                  # Run browser in background
PLAYWRIGHT_TIMEOUT=30000                  # 30 second timeout
PLAYWRIGHT_USER_AGENT="UltraThink/1.0"    # Custom user agent

# Rate limiting (to avoid being blocked)
SCRAPER_MAX_REQUESTS_PER_MINUTE=30
SCRAPER_RETRY_ATTEMPTS=3
SCRAPER_CACHE_TTL=300                     # Cache for 5 minutes
```

### Customize Scraping Sources

Edit `tools/playwright_mcp.py` to add your preferred sources:

```python
# In PlaywrightMarketScraper
def get_crypto_news(self, symbol: str, sources: List[str] = None):
    if sources is None:
        sources = [
            "https://coindesk.com",
            "https://cointelegraph.com",
            "https://decrypt.co",
            # Add your preferred news sources
        ]
    # ...
```

---

## Performance Considerations

### Caching

Playwright scraping can be slow. Implement caching:

```python
from functools import lru_cache
from datetime import datetime, timedelta

class PlaywrightMarketScraper:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    @lru_cache(maxsize=100)
    def get_crypto_news_cached(self, symbol: str, limit: int = 10):
        cache_key = f"news_{symbol}_{limit}"
        cached = self.cache.get(cache_key)

        if cached and (datetime.now() - cached['timestamp']).seconds < self.cache_ttl:
            return cached['data']

        # Fetch fresh data
        data = self.get_crypto_news(symbol, limit)
        self.cache[cache_key] = {'data': data, 'timestamp': datetime.now()}
        return data
```

### Parallel Scraping

Scrape multiple sources concurrently:

```python
import concurrent.futures

def scrape_multiple_exchanges(symbol: str):
    exchanges = ["Binance", "Coinbase", "Kraken", "Gemini"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(scrape_exchange, exchange, symbol): exchange
            for exchange in exchanges
        }

        results = {}
        for future in concurrent.futures.as_completed(futures):
            exchange = futures[future]
            try:
                results[exchange] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"Failed to scrape {exchange}: {e}")

    return results
```

### Rate Limiting

Avoid being blocked by implementing rate limits:

```python
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    def wait_if_needed(self):
        now = time.time()

        # Remove old calls outside the period
        while self.calls and self.calls[0] < now - self.period:
            self.calls.popleft()

        # Wait if at limit
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.calls.append(now)

# Usage
limiter = RateLimiter(max_calls=30, period=60.0)  # 30 calls per minute

def scrape_with_rate_limit(url):
    limiter.wait_if_needed()
    # ... perform scraping
```

---

## Testing

### Unit Tests

Create `tests/test_playwright_mcp.py`:

```python
import pytest
from tools.playwright_mcp import PlaywrightMarketScraper, PlaywrightDataEnricher

class TestPlaywrightMCP:
    def test_scraper_initialization(self):
        scraper = PlaywrightMarketScraper()
        assert scraper is not None

    def test_get_crypto_news(self):
        scraper = PlaywrightMarketScraper()
        news = scraper.get_crypto_news("BTC", limit=5)
        assert len(news) <= 5
        assert all(hasattr(n, 'title') for n in news)

    def test_get_exchange_prices(self):
        scraper = PlaywrightMarketScraper()
        prices = scraper.get_exchange_prices("BTC-USD")
        assert isinstance(prices, dict)
        assert len(prices) > 0

    def test_data_enricher(self):
        enricher = PlaywrightDataEnricher()
        context = {"price": 50000, "rsi": 65}
        enriched = enricher.enrich_market_context(context, "BTC")
        assert 'recent_news' in enriched
        assert 'social_sentiment' in enriched
```

### Integration Tests

Test with real backtest:

```bash
# Run test with Playwright enrichment
python run_backtest.py \
  --symbol BTC-USD \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --use-playwright \
  --capital 100000
```

---

## Troubleshooting

### Issue: "Playwright not found"

**Solution:** Playwright MCP is built into Claude Code. Ensure you're running this within Claude Code, not standalone Python.

### Issue: Scraping returns empty results

**Solutions:**
1. Check network connection
2. Verify target website is accessible
3. Inspect browser console for errors
4. Try increasing timeout values
5. Check if website has anti-scraping measures

### Issue: Rate limiting / IP blocked

**Solutions:**
1. Reduce scraping frequency
2. Implement rate limiting (see above)
3. Add random delays between requests
4. Use rotating user agents
5. Consider using API endpoints instead

### Issue: Stale data

**Solutions:**
1. Clear cache: `scraper.cache.clear()`
2. Reduce `SCRAPER_CACHE_TTL` in config
3. Force fresh fetch by bypassing cache

---

## Examples

### Complete Example: News-Aware Backtesting

```python
from backtesting.backtest_engine import BacktestEngine
from tools.playwright_mcp import PlaywrightMarketScraper, PlaywrightDataEnricher
import logging

logging.basicConfig(level=logging.INFO)

# Initialize components
engine = BacktestEngine(
    symbol="BTC-USD",
    start_date="2024-01-01",
    end_date="2024-06-01",
    initial_capital=100000,
    use_openai=False  # Use mock agents
)

scraper = PlaywrightMarketScraper()
enricher = PlaywrightDataEnricher()

# Load data
engine.load_data()

# Custom backtest loop with news awareness
for i, row in engine.data.iterrows():
    if i < engine.skip_days:
        continue

    # Standard context
    context = engine.data_fetcher.get_market_context(i)

    # Enrich with web data every hour (to avoid rate limits)
    if i % 60 == 0:  # Assuming minute-level data
        enriched_context = enricher.enrich_market_context(context, "BTC")
    else:
        enriched_context = context

    # Get agent recommendations with enriched data
    mr_result = engine.call_mr_sr_agent(enriched_context, row['Date'])
    ers_result = engine.call_ers_agent(mr_result, row['Date'])

    # Execute trade
    if ers_result['decision'] == 'APPROVE':
        action = mr_result['action']
        engine.portfolio.execute_trade(action, row['Close'], 0.2)

# Generate report
report = engine.generate_report()
print(f"Final Return: {report['metrics']['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {report['metrics']['sharpe_ratio']:.2f}")
```

---

## API Reference

### PlaywrightMarketScraper

```python
class PlaywrightMarketScraper:
    def __init__()

    def get_crypto_news(symbol: str, limit: int = 10) -> List[MarketNewsItem]

    def get_exchange_prices(symbol: str, exchanges: List[str] = None) -> Dict[str, PriceData]

    def capture_trading_dashboard(url: str, output_path: str, full_page: bool = True) -> str

    def monitor_twitter_sentiment(symbol: str, duration_minutes: int = 60) -> Dict[str, Any]

    def test_trading_interface(exchange_url: str, test_actions: List[str]) -> Dict[str, Any]
```

### PlaywrightDataEnricher

```python
class PlaywrightDataEnricher:
    def __init__()

    def enrich_market_context(context: Dict[str, Any], symbol: str) -> Dict[str, Any]

    def create_visual_report(backtest_results: Dict[str, Any], output_dir: str) -> List[str]
```

---

## Best Practices

1. **Cache aggressively** - Web scraping is slow
2. **Respect rate limits** - Don't get IP banned
3. **Handle errors gracefully** - Websites change, scraping breaks
4. **Use headless mode** - Faster and less resource-intensive
5. **Validate scraped data** - Sanity check prices, dates, etc.
6. **Combine with APIs** - Use Playwright only when API unavailable
7. **Monitor performance** - Log scraping times and success rates
8. **Test regularly** - Websites change their HTML structure

---

## Future Enhancements

- [ ] Image analysis of charts (pattern recognition)
- [ ] Automated order execution via exchange UIs
- [ ] Real-time WebSocket monitoring
- [ ] Multi-language news sentiment (not just English)
- [ ] Video content analysis (YouTube, etc.)
- [ ] Deeper Reddit/Discord sentiment analysis
- [ ] Automated alert system based on scraped data
- [ ] Historical news archive scraping
- [ ] Correlate news events with price movements

---

## Support

For issues or questions:
- Check Claude Code Playwright MCP docs: https://docs.claude.com
- Review BUG_REPORT.md for known issues
- See CLAUDE.md for development guidelines

**Pro Tip:** Start with mock data in `playwright_mcp.py`, then gradually implement real scraping as needed. The infrastructure is ready!
