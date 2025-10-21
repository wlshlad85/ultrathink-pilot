# Hugging Face MCP Integration for UltraThink

## Overview

UltraThink now supports **ML model research and integration** via Hugging Face MCP (Model Context Protocol), enabling access to 500K+ models, 100K+ datasets, and cutting-edge research papers for cryptocurrency trading with reinforcement learning.

### What You Can Do

- 🤖 **Search ML models** for trading, sentiment analysis, and RL benchmarks
- 📊 **Access datasets** with financial news, crypto prices, and market data
- 📚 **Research papers** on reinforcement learning and algorithmic trading
- 🎯 **Benchmark your PPO agent** against state-of-the-art models
- 💭 **Add sentiment analysis** to your trading pipeline
- 🔍 **Discover pre-trained models** for feature engineering

---

## Quick Start

### 1. Hugging Face MCP is Already Installed

According to `USEFUL_MCPS.md`, the Hugging Face MCP server is already configured!

**Verify installation:**
```bash
# In Claude Code
/mcp

# Look for: huggingface (connected)
```

**If not installed:**
```bash
claude mcp add huggingface -- cmd /c npx -y @llmindset/hf-mcp-server
```

### 2. Use the Python Interface

```python
from tools.huggingface_mcp import HuggingFaceClient, HFTradingResearch, HFDataEnricher

# Search for crypto trading models
client = HuggingFaceClient()
models = client.search_models("bitcoin trading PPO", limit=5)

for model in models:
    print(f"{model.model_id}: {model.downloads} downloads")
    print(f"  Tags: {', '.join(model.tags)}")
    print(f"  Task: {model.task}")
```

### 3. Research Trading Models

```python
# Use specialized research assistant
research = HFTradingResearch()

# Find crypto-specific trading models
trading_models = research.find_crypto_trading_models()
print(f"Found {len(trading_models)} crypto trading models")

# Find sentiment analysis models
sentiment_models = research.find_sentiment_models()

# Find financial datasets
datasets = research.find_financial_datasets()

# Find latest RL trading papers
papers = research.find_rl_trading_papers()

# Generate comprehensive report
report = research.generate_research_report()
print(report["recommendations"])
```

---

## Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    UltraThink System                         │
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ Python Interface │         │ Claude Code MCP   │         │
│  │ (hf_mcp.py)      │────────▶│ HF Tools          │         │
│  └──────────────────┘         └──────────────────┘         │
│           │                            │                     │
│           │                            ▼                     │
│           │                   ┌──────────────────┐          │
│           │                   │  Hugging Face    │          │
│           │                   │  Hub API         │          │
│           │                   └──────────────────┘          │
│           │                            │                     │
│           ▼                            ▼                     │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Research        │         │  Models/Datasets  │         │
│  │  (papers, models)│◀────────│  (download)       │         │
│  └──────────────────┘         └──────────────────┘         │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ PPO Agent        │                                       │
│  │ (benchmark)      │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **HuggingFaceClient** (`tools/huggingface_mcp.py`)
   - High-level Python interface for HF Hub
   - Search models, datasets, and papers
   - Get detailed model metadata
   - Download models and datasets

2. **HFTradingResearch** (`tools/huggingface_mcp.py`)
   - Specialized research assistant for trading
   - Find crypto/stock trading models
   - Discover sentiment analysis models
   - Locate financial datasets
   - Search RL trading papers

3. **HFDataEnricher** (`tools/huggingface_mcp.py`)
   - Integrate HF models into trading pipeline
   - Add sentiment analysis to market context
   - Use pre-trained models for feature engineering
   - Compatible with existing DataFetcher output

4. **Claude Code Hugging Face MCP**
   - Built-in HF Hub integration
   - Available via MCP server
   - 11 out of 13 tools enabled

---

## Use Cases

### 1. Benchmark Against CryptoTrader-LM

Compare your PPO agent with state-of-the-art:

```python
from tools.huggingface_mcp import HuggingFaceClient

client = HuggingFaceClient()

# Find CryptoTrader-LM model
model = client.get_model_info("agarkovv/CryptoTrader-LM")
print(f"CryptoTrader-LM: {model.description}")

# Download for benchmarking
success, msg = client.download_model(
    "agarkovv/CryptoTrader-LM",
    local_dir="./models/cryptotrader_lm"
)

if success:
    # Use download_cryptotrader_lm.py for actual implementation
    print("Ready to benchmark!")
```

### 2. Add Sentiment Analysis to Trading Pipeline

Enhance your MR-SR agent with news sentiment:

```python
from tools.huggingface_mcp import HFDataEnricher
from backtesting.data_fetcher import DataFetcher

# Get standard market data
fetcher = DataFetcher("BTC-USD", "2024-01-01", "2024-12-01")
context = fetcher.get_market_context(100)

# Enrich with sentiment
enricher = HFDataEnricher()
enricher.load_sentiment_model("ElKulako/cryptobert")

news = [
    "Bitcoin surges past $50K as institutions increase allocation",
    "Crypto market sees record trading volumes"
]

enriched_context = enricher.enrich_market_context(context, news)

# Now includes sentiment scores
print(enriched_context["hf_news_sentiment"])
# {
#   "average_score": 0.78,
#   "bullish_ratio": 1.0,
#   "total_analyzed": 2
# }
```

### 3. Research Latest RL Trading Papers

Stay updated with cutting-edge research:

```python
from tools.huggingface_mcp import HFTradingResearch

research = HFTradingResearch()

# Find latest papers
papers = research.find_rl_trading_papers()

print("Top RL Trading Papers:\n")
for paper in papers[:5]:
    print(f"📄 {paper.title}")
    print(f"   arXiv: {paper.arxiv_id}")
    print(f"   Upvotes: {paper.upvotes}")
    print(f"   Tags: {', '.join(paper.tags[:3])}\n")
```

### 4. Find Financial Datasets for Training

Enhance your training data:

```python
from tools.huggingface_mcp import HuggingFaceClient

client = HuggingFaceClient()

# Search for crypto/financial datasets
datasets = client.search_datasets("cryptocurrency price OR bitcoin OHLCV", limit=10)

for ds in datasets:
    print(f"\n{ds.dataset_id}")
    print(f"  Rows: {ds.num_rows:,}")
    print(f"  Downloads: {ds.downloads}")
    print(f"  Size: {ds.size_category}")
    print(f"  Description: {ds.description}")
```

### 5. Generate Comprehensive Research Report

Get actionable recommendations:

```python
from tools.huggingface_mcp import HFTradingResearch

research = HFTradingResearch()
report = research.generate_research_report()

print("Trading Models:")
for model in report["trading_models"]:
    print(f"  - {model.model_id}")

print("\nSentiment Models:")
for model in report["sentiment_models"]:
    print(f"  - {model.model_id}")

print("\nRecommendations:")
recs = report["recommendations"]
print(f"  Benchmark against: {recs['benchmark_against']}")
print(f"  Add sentiment from: {recs['add_sentiment_from']}")
print(f"  Enhance data with: {recs['enhance_data_with']}")
print(f"  Read papers: {', '.join(recs['read_papers'])}")
```

---

## Integration with Existing Code

### Modify MR-SR Agent to Use Sentiment

Edit `agents/mr_sr.py` to incorporate HF sentiment:

```python
# In agents/mr_sr.py
from tools.huggingface_mcp import HFDataEnricher

def generate_recommendation(context: Dict[str, Any]) -> Dict[str, Any]:
    # Original indicators
    rsi = context.get('rsi', 50)
    macd = context.get('macd_signal', 'NEUTRAL')

    # NEW: HF-powered sentiment
    hf_sentiment = context.get('hf_news_sentiment', {})
    sentiment_score = hf_sentiment.get('average_score', 0.5)
    bullish_ratio = hf_sentiment.get('bullish_ratio', 0.5)

    # Adjust strategy based on sentiment + technicals
    if sentiment_score > 0.7 and bullish_ratio > 0.6 and rsi < 70:
        action = "BUY"
        confidence = min(sentiment_score, 0.9)
        reasoning = f"Strong bullish sentiment ({sentiment_score:.2f}) + favorable RSI"
    elif sentiment_score < 0.3 and bullish_ratio < 0.4 and rsi > 30:
        action = "SELL"
        confidence = min(1 - sentiment_score, 0.9)
        reasoning = f"Bearish sentiment ({sentiment_score:.2f}) + technical weakness"
    else:
        action = "HOLD"
        confidence = 0.5
        reasoning = "Mixed signals from sentiment and technicals"

    return {
        "action": action,
        "confidence": confidence,
        "reasoning": reasoning,
        "sentiment_score": sentiment_score,
        # ... rest of recommendation
    }
```

### Enhance Backtesting with HF Models

Modify `backtesting/backtest_engine.py`:

```python
# In BacktestEngine.__init__
from tools.huggingface_mcp import HFDataEnricher

self.hf_enricher = HFDataEnricher()
self.use_hf_sentiment = use_hf_sentiment  # New parameter

# In BacktestEngine.run()
for i, row in self.data.iterrows():
    if i < self.skip_days:
        continue

    context = self.data_fetcher.get_market_context(i)

    # Enrich with HF sentiment if enabled
    if self.use_hf_sentiment:
        # Fetch news for this date (could use financial-news-multisource dataset)
        news = self._fetch_news_for_date(row['Date'])
        context = self.hf_enricher.enrich_market_context(context, news)

    # Continue with agent calls...
    mr_result = self._call_agent('mr_sr', context, row['Date'])
    # ...
```

---

## Key Models and Resources

### Top Trading Models on Hugging Face

1. **agarkovv/CryptoTrader-LM** ⭐
   - Crypto trading (BTC/ETH)
   - Trained on 2022-2024 data
   - Actions: Buy, Sell, Hold
   - FinNLP @ COLING-2025

2. **Adilbai/stock-trading-rl-agent**
   - PPO for stock trading
   - Direct comparison to your agent
   - PyTorch implementation

3. **ElKulako/cryptobert**
   - Sentiment analysis
   - 3.2M crypto social media posts
   - BERT-based

4. **Taoshi/model_v4**
   - Multi-modal features
   - Bitcoin OHLCV + sentiment
   - Open interest + funding rates

### Top Datasets

1. **Brianferrell787/financial-news-multisource**
   - 47M+ rows (1990-2024)
   - Multi-source financial news
   - Perfect for sentiment analysis

2. **Josephgflowers/Finance-Instruct-500k**
   - 500K+ financial reasoning examples
   - Sentiment + multi-turn conversations
   - NLP training data

### Top Papers (2024-2025)

1. **Meta-RL-Crypto** (arXiv:2509.09751)
   - Most advanced crypto RL framework
   - Transformer + meta-learning
   - Self-improving trading agent

2. **FLAG-Trader** (2025)
   - LLM + PPO fusion
   - Gradient-based RL

3. **Deep RL for Crypto: Addressing Backtest Overfitting** (arXiv:2209.05559)
   - Critical for validation
   - Prevent overfitting

See `HUGGINGFACE_RL_RESOURCES.md` for complete list.

---

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Hugging Face Configuration
HF_TOKEN=hf_xxxxxxxxxxxxx              # Optional: For private models
HF_CACHE_DIR=./models/hf_cache         # Model download cache
HF_DATASETS_CACHE=./data/hf_cache      # Dataset cache

# Sentiment Analysis
HF_SENTIMENT_MODEL=ElKulako/cryptobert
HF_SENTIMENT_BATCH_SIZE=32

# Rate Limiting
HF_MAX_REQUESTS_PER_MINUTE=10
```

### Policy Configuration

HF tools are already registered in `policy/tools.yaml`:

```yaml
hf_search_models:
  class: A  # Read-only
  timeout_sec: 15
  rate_limit_per_minute: 10

hf_sentiment_analysis:
  class: A
  timeout_sec: 10
  rate_limit_per_minute: 30

hf_download_model:
  class: B  # Writes to disk (sandbox)
  timeout_sec: 60
  rate_limit_per_minute: 2
```

---

## Best Practices

### 1. Caching

HF models and datasets are large. Use caching:

```python
import os
os.environ["HF_HOME"] = "./models/hf_cache"

from transformers import AutoModelForSequenceClassification

# Will cache automatically
model = AutoModelForSequenceClassification.from_pretrained(
    "ElKulako/cryptobert",
    cache_dir="./models/hf_cache"
)
```

### 2. Rate Limiting

Respect HF API limits:

```python
from tools.huggingface_mcp import HuggingFaceClient
import time

client = HuggingFaceClient()

# Batch searches with delays
queries = ["bitcoin", "ethereum", "cardano"]
results = []

for query in queries:
    models = client.search_models(f"{query} trading")
    results.extend(models)
    time.sleep(6)  # 10 requests/min = 6 sec between calls
```

### 3. Model Selection

Choose appropriate models for your use case:

| Task | Recommended Model | Purpose |
|------|------------------|---------|
| Sentiment | ElKulako/cryptobert | Crypto news sentiment |
| Trading | agarkovv/CryptoTrader-LM | Benchmark comparison |
| General NLP | FinGPT models | Financial text understanding |

### 4. Validation

Always validate model outputs:

```python
enricher = HFDataEnricher()
sentiment = enricher.analyze_news_sentiment("Bitcoin crashes!")

# Sanity check
assert 0 <= sentiment["score"] <= 1
assert sentiment["label"] in ["BULLISH", "BEARISH", "NEUTRAL"]
```

### 5. Benchmarking

Compare fairly with your PPO agent:

```python
# Use same data split
# Use same evaluation metrics
# Account for different input formats
# Report Sharpe, returns, drawdown, win rate
```

---

## Testing

### Unit Tests

Create `tests/test_huggingface_mcp.py`:

```python
import pytest
from tools.huggingface_mcp import (
    HuggingFaceClient,
    HFTradingResearch,
    HFDataEnricher
)

class TestHuggingFaceMCP:
    def test_client_initialization(self):
        client = HuggingFaceClient()
        assert client is not None

    def test_search_models(self):
        client = HuggingFaceClient()
        models = client.search_models("bitcoin trading", limit=3)
        assert len(models) <= 3
        assert all(hasattr(m, 'model_id') for m in models)

    def test_search_datasets(self):
        client = HuggingFaceClient()
        datasets = client.search_datasets("financial news", limit=2)
        assert len(datasets) <= 2
        assert all(hasattr(d, 'dataset_id') for d in datasets)

    def test_research_assistant(self):
        research = HFTradingResearch()
        models = research.find_crypto_trading_models()
        assert isinstance(models, list)

    def test_data_enricher(self):
        enricher = HFDataEnricher()
        context = {"price": 50000, "rsi": 65}
        news = ["Bitcoin breaks resistance"]
        enriched = enricher.enrich_market_context(context, news)
        assert 'hf_news_sentiment' in enriched
```

Run tests:
```bash
pytest tests/test_huggingface_mcp.py -v
```

---

## Troubleshooting

### Issue: "HF MCP not found"

**Solution:**
```bash
# Check if installed
claude mcp list

# If not installed
claude mcp add huggingface -- cmd /c npx -y @llmindset/hf-mcp-server

# Verify
/mcp  # In Claude Code
```

### Issue: Model download fails

**Solutions:**
1. Check internet connection
2. Verify model ID is correct
3. Check disk space
4. Use authentication token for private models:
   ```python
   os.environ["HF_TOKEN"] = "hf_xxxxx"
   ```

### Issue: Rate limiting errors

**Solutions:**
1. Reduce request frequency
2. Implement proper delays
3. Cache results
4. Use batch operations

### Issue: Out of memory when loading models

**Solutions:**
1. Use smaller models
2. Load on CPU: `device_map="cpu"`
3. Use 8-bit quantization:
   ```python
   model = AutoModel.from_pretrained(
       model_id,
       load_in_8bit=True,
       device_map="auto"
   )
   ```

---

## Examples

### Complete Example: Sentiment-Enhanced Backtesting

```python
#!/usr/bin/env python3
"""
Backtest with HF sentiment analysis integration
"""

from backtesting.backtest_engine import BacktestEngine
from tools.huggingface_mcp import HFDataEnricher
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Initialize backtest
    engine = BacktestEngine(
        symbol="BTC-USD",
        start_date="2024-01-01",
        end_date="2024-06-01",
        initial_capital=100000,
        use_openai=False
    )

    # Initialize HF enricher
    enricher = HFDataEnricher()
    enricher.load_sentiment_model("ElKulako/cryptobert")

    # Load data
    engine.load_data()

    # Run backtest with sentiment
    for i, row in engine.data.iterrows():
        if i < engine.skip_days:
            continue

        # Get standard context
        context = engine.data_fetcher.get_market_context(i)

        # Fetch news for this date (mock for demo)
        news = [
            f"Market update for {row['Date']}",
            f"Bitcoin trading at ${row['Close']:,.2f}"
        ]

        # Enrich with sentiment
        enriched = enricher.enrich_market_context(context, news)

        # Get agent decisions with enriched data
        mr_result = engine.call_mr_sr_agent(enriched, row['Date'])
        ers_result = engine.call_ers_agent(mr_result, row['Date'])

        # Execute trade
        if ers_result['decision'] == 'APPROVE':
            action = mr_result['action']
            engine.portfolio.execute_trade(action, row['Close'], 0.2)

    # Generate report
    report = engine.generate_report()

    print("\n=== Sentiment-Enhanced Backtest Results ===")
    print(f"Total Return: {report['metrics']['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {report['metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {report['metrics']['max_drawdown_pct']:.2f}%")

if __name__ == "__main__":
    main()
```

---

## API Reference

### HuggingFaceClient

```python
class HuggingFaceClient:
    def search_models(query: str, task: ModelTask = None, limit: int = 10) -> List[HFModel]
    def search_datasets(query: str, size: str = None, limit: int = 10) -> List[HFDataset]
    def search_papers(query: str, sort: str = "trending", limit: int = 10) -> List[HFPaper]
    def get_model_info(model_id: str) -> Optional[HFModel]
    def download_model(model_id: str, local_dir: str = "./models") -> Tuple[bool, str]
```

### HFTradingResearch

```python
class HFTradingResearch:
    def find_crypto_trading_models() -> List[HFModel]
    def find_sentiment_models() -> List[HFModel]
    def find_financial_datasets() -> List[HFDataset]
    def find_rl_trading_papers() -> List[HFPaper]
    def generate_research_report() -> Dict[str, Any]
```

### HFDataEnricher

```python
class HFDataEnricher:
    def load_sentiment_model(model_id: str = "ElKulako/cryptobert") -> None
    def analyze_news_sentiment(news_text: str) -> Dict[str, Any]
    def enrich_market_context(context: Dict, news_items: List[str]) -> Dict[str, Any]
```

---

## Next Steps

### Week 1: Setup & Exploration
1. ✅ Install HF MCP server (already done!)
2. ✅ Create Python integration tool (huggingface_mcp.py)
3. [ ] Test model search functionality
4. [ ] Explore CryptoTrader-LM model
5. [ ] Review top RL trading papers

### Week 2: Sentiment Integration
6. [ ] Download CryptoBERT sentiment model
7. [ ] Add sentiment to state space
8. [ ] Test backtesting with sentiment
9. [ ] Compare performance with/without sentiment

### Week 3: Benchmarking
10. [ ] Download CryptoTrader-LM
11. [ ] Create benchmark script
12. [ ] Compare with your PPO agent
13. [ ] Document performance differences

### Week 4: Advanced Features
14. [ ] Explore financial news datasets
15. [ ] Implement ensemble with HF models
16. [ ] Research meta-learning approaches
17. [ ] Publish findings

---

## Resources

- **Project Docs**: See `HUGGINGFACE_RL_RESOURCES.md` for research findings
- **Model Download**: See `download_cryptotrader_lm.py` for example
- **Research Papers**: See `RESEARCH_PAPERS_HUGGINGFACE.md`
- **Development**: See `CLAUDE.md` for contribution guidelines
- **MCP Guide**: See `USEFUL_MCPS.md` for MCP overview

---

**Generated**: October 21, 2025
**Project**: ultrathink-pilot (Bitcoin RL Trading with PPO)
**Status**: Hugging Face MCP Integration Complete
**Version**: 1.0

**Ready to Use!** Start with:
```python
from tools.huggingface_mcp import HuggingFaceClient
client = HuggingFaceClient()
models = client.search_models("bitcoin trading")
```
