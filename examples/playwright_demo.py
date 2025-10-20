#!/usr/bin/env python3
"""
Playwright MCP Demo for UltraThink

Demonstrates how to use browser automation for enhanced trading insights.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.playwright_mcp import PlaywrightMarketScraper, PlaywrightDataEnricher
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def demo_news_scraping():
    """Demo 1: Scraping crypto news"""
    print("\n" + "=" * 70)
    print("DEMO 1: Crypto News Scraping")
    print("=" * 70)

    scraper = PlaywrightMarketScraper()

    symbols = ["BTC", "ETH", "SOL"]

    for symbol in symbols:
        print(f"\n📰 Latest {symbol} News:")
        news = scraper.get_crypto_news(symbol, limit=5)

        for i, item in enumerate(news, 1):
            sentiment_emoji = {
                "BULLISH": "📈",
                "BEARISH": "📉",
                "NEUTRAL": "➡️"
            }.get(item.sentiment, "❓")

            print(f"  {i}. {sentiment_emoji} {item.title}")
            print(f"     Source: {item.source} | {item.timestamp.strftime('%Y-%m-%d %H:%M')}")


def demo_price_comparison():
    """Demo 2: Exchange price comparison"""
    print("\n" + "=" * 70)
    print("DEMO 2: Exchange Price Comparison")
    print("=" * 70)

    scraper = PlaywrightMarketScraper()

    symbols = ["BTC-USD", "ETH-USD"]

    for symbol in symbols:
        print(f"\n💹 {symbol} Prices:")
        prices = scraper.get_exchange_prices(symbol)

        prices_list = [(ex, data.price, data.volume_24h) for ex, data in prices.items()]
        prices_list.sort(key=lambda x: x[1])

        for exchange, price, volume in prices_list:
            volume_str = f"${volume/1e6:.1f}M" if volume else "N/A"
            print(f"  {exchange:12s}: ${price:>10,.2f}  (Vol: {volume_str})")

        # Calculate spread
        lowest = prices_list[0][1]
        highest = prices_list[-1][1]
        spread_pct = ((highest - lowest) / lowest) * 100

        print(f"\n  📊 Spread: {spread_pct:.2f}%")
        if spread_pct > 0.5:
            print(f"  ⚡ Arbitrage opportunity detected!")


def demo_sentiment_analysis():
    """Demo 3: Social media sentiment"""
    print("\n" + "=" * 70)
    print("DEMO 3: Social Media Sentiment Analysis")
    print("=" * 70)

    scraper = PlaywrightMarketScraper()

    symbols = ["BTC", "ETH", "SOL"]

    for symbol in symbols:
        print(f"\n🐦 {symbol} Twitter Sentiment:")
        sentiment = scraper.monitor_twitter_sentiment(symbol)

        score = sentiment['sentiment_score']

        # Determine sentiment category
        if score > 0.7:
            category = "🟢 VERY BULLISH"
        elif score > 0.55:
            category = "🟢 BULLISH"
        elif score > 0.45:
            category = "⚪ NEUTRAL"
        elif score > 0.3:
            category = "🔴 BEARISH"
        else:
            category = "🔴 VERY BEARISH"

        print(f"  Overall Sentiment: {category} ({score:.2f}/1.00)")
        print(f"  Total Mentions: {sentiment['total_mentions']}")

        breakdown = sentiment['sentiment_breakdown']
        total = sum(breakdown.values())
        print(f"  Breakdown:")
        print(f"    Bullish:  {breakdown['bullish']:3d} ({breakdown['bullish']/total*100:.0f}%)")
        print(f"    Bearish:  {breakdown['bearish']:3d} ({breakdown['bearish']/total*100:.0f}%)")
        print(f"    Neutral:  {breakdown['neutral']:3d} ({breakdown['neutral']/total*100:.0f}%)")

        print(f"  Top Keywords: {', '.join(sentiment['top_keywords'])}")
        print(f"  Influencer Mentions: {sentiment['influencer_mentions']}")


def demo_context_enrichment():
    """Demo 4: Enriching backtest context"""
    print("\n" + "=" * 70)
    print("DEMO 4: Market Context Enrichment")
    print("=" * 70)

    # Simulate standard backtest context
    standard_context = {
        "asset": "BTC-USD",
        "price": 50000.0,
        "rsi": 65.5,
        "macd_signal": "BULLISH",
        "atr14": 1500.0,
        "volume_ratio": 1.2
    }

    print("\n📊 Standard Context (from DataFetcher):")
    for key, value in standard_context.items():
        print(f"  {key:15s}: {value}")

    # Enrich with web data
    enricher = PlaywrightDataEnricher()
    enriched = enricher.enrich_market_context(standard_context, "BTC")

    print("\n✨ Enriched Context (with Playwright MCP):")
    print(f"  Original keys: {len(standard_context)}")
    print(f"  Enriched keys: {len(enriched)}")

    print("\n  New data added:")
    print(f"    - Recent news: {len(enriched.get('recent_news', []))} articles")
    print(f"    - Exchange prices: {len(enriched.get('exchange_prices', {}))} exchanges")
    print(f"    - Social sentiment: {enriched.get('social_sentiment', {}).get('score', 0):.2f}")

    # Show how this improves decision-making
    print("\n🤖 Impact on Trading Decision:")
    print("  Standard approach: Only technical indicators")
    print("  Enhanced approach: Technical + News + Sentiment + Multi-exchange")


def demo_trading_signal():
    """Demo 5: Generate enhanced trading signal"""
    print("\n" + "=" * 70)
    print("DEMO 5: Enhanced Trading Signal Generation")
    print("=" * 70)

    scraper = PlaywrightMarketScraper()

    symbol = "BTC"
    print(f"\n🎯 Generating Trading Signal for {symbol}...\n")

    # Gather data
    news = scraper.get_crypto_news(symbol, limit=10)
    prices = scraper.get_exchange_prices(f"{symbol}-USD")
    sentiment = scraper.monitor_twitter_sentiment(symbol)

    # Analyze
    bullish_news = sum(1 for n in news if n.sentiment == "BULLISH")
    bearish_news = sum(1 for n in news if n.sentiment == "BEARISH")
    sentiment_score = sentiment['sentiment_score']

    prices_list = [data.price for data in prices.values()]
    avg_price = sum(prices_list) / len(prices_list)

    # Generate signal
    print("📊 Analysis:")
    print(f"  News Sentiment: {bullish_news} bullish, {bearish_news} bearish")
    print(f"  Social Sentiment: {sentiment_score:.2f}/1.00")
    print(f"  Average Price: ${avg_price:,.2f}")

    # Decision logic
    signals = []
    if sentiment_score > 0.7:
        signals.append("BUY (strong social sentiment)")
    elif sentiment_score < 0.3:
        signals.append("SELL (weak social sentiment)")

    if bullish_news > bearish_news * 2:
        signals.append("BUY (bullish news)")
    elif bearish_news > bullish_news * 2:
        signals.append("SELL (bearish news)")

    print(f"\n⚡ Trading Signals:")
    if signals:
        for signal in signals:
            print(f"  - {signal}")

        # Overall recommendation
        buy_signals = sum(1 for s in signals if "BUY" in s)
        sell_signals = sum(1 for s in signals if "SELL" in s)

        if buy_signals > sell_signals:
            print(f"\n  🟢 RECOMMENDATION: BUY")
        elif sell_signals > buy_signals:
            print(f"\n  🔴 RECOMMENDATION: SELL")
        else:
            print(f"\n  ⚪ RECOMMENDATION: HOLD")
    else:
        print(f"  ⚪ RECOMMENDATION: HOLD (insufficient signals)")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("UltraThink Playwright MCP Integration Demo")
    print("=" * 70)

    try:
        demo_news_scraping()
        demo_price_comparison()
        demo_sentiment_analysis()
        demo_context_enrichment()
        demo_trading_signal()

        print("\n" + "=" * 70)
        print("[SUCCESS] All demos completed successfully!")
        print("=" * 70)

        print("\n[NEXT STEPS]")
        print("  1. Review PLAYWRIGHT_MCP.md for full documentation")
        print("  2. Integrate scraper into your backtesting workflow")
        print("  3. Modify MR-SR agent to use enriched context")
        print("  4. Run enhanced backtest: python run_backtest.py --use-playwright")
        print("\n[TIP] Start with mock data, then implement real scraping gradually")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
