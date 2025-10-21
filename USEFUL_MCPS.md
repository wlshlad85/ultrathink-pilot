# 🚀 Useful MCP Servers for Bitcoin RL Trading

**Project**: ultrathink-pilot (Bitcoin RL Trading with PPO)
**Date**: October 20, 2025
**Purpose**: MCP servers to enhance ML experiment tracking, data analysis, and trading development

---

## 📊 Installation Priority

Install in this order for maximum value:

1. **CCXT** ✅ (Already installed!)
2. **Pandas** (Data analysis)
3. **GitHub** (Version control)
4. **SQLite** (Database queries)
5. **Jupyter** (Notebooks)
6. **Postgres** (When scaling)
7. **Freqtrade** (Live trading prep)

---

## 🎯 Top Priority - Install These First

### 1. ✅ CCXT (Cryptocurrency Exchange Data) - INSTALLED

**Status**: ✓ Connected
**What it does**: Access to 100+ cryptocurrency exchanges for real-time and historical data

**Installation**:
```bash
claude mcp add ccxt -- npx -y @mcpfun/mcp-server-ccxt
```

**Use cases**:
- Get real-time Bitcoin prices from multiple exchanges
- Fetch historical OHLCV data for training
- Access order book depth for better features
- Compare market data across exchanges
- Detect market regimes in real-time

**Example queries**:
```
"Get current BTC-USD price from Binance"
"Fetch 1-hour OHLCV data for Bitcoin from the last 30 days"
"Compare Bitcoin prices across Binance, Coinbase, and Kraken"
"What's the order book depth for BTC-USD on Binance?"
```

**Authentication**: None required!

---

### 2. ✅ Pandas MCP (Data Analysis) - INSTALLED

**Status**: ✓ Connected
**What it does**: Powerful pandas operations directly through Claude for analyzing your experiments

**Installation** (completed):
```bash
# Step 1: Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Step 2: Clone and setup pandas-mcp-server
git clone https://github.com/marlonluo2018/pandas-mcp-server.git ~/pandas-mcp-server
cd ~/pandas-mcp-server
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Step 3: Configure with Claude
claude mcp add pandas -- wsl bash -c "cd ~/pandas-mcp-server && source .venv/bin/activate && python3 server.py"
```

**Use cases**:
- Analyze your 10 experiments to find best hyperparameters
- Statistical analysis of returns across market regimes
- Compare training runs with complex queries
- Feature importance analysis
- Correlation studies between metrics

**Example queries**:
```
"Load my ml_experiments.db and analyze which experiments had the best returns"
"Compare training curves across all experiments"
"Find correlations between hyperparameters and final returns"
"Show me statistical summary of all episode returns"
```

**Authentication**: None required!

---

### 3. ✅ GitHub MCP (Version Control) - INSTALLED

**Status**: ✓ Connected
**What it does**: Create PRs, manage issues, review code directly from Claude

**Installation**:
```bash
# First, create a GitHub Personal Access Token:
# 1. Go to: https://github.com/settings/tokens/new
# 2. Name: "Claude MCP Server"
# 3. Scopes: repo, read:org, user
# 4. Copy the token

# Then install:
claude mcp add github \
  --env GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here \
  -- npx -y @modelcontextprotocol/server-github
```

**Use cases**:
- Create PRs for new features
- Track experiment results as GitHub issues
- Review code changes
- Automate experiment documentation
- Link experiments to specific commits

**Example queries**:
```
"Create a GitHub issue documenting Experiment #10 results"
"Create a PR with the latest training improvements"
"Show me recent commits in this repository"
"Review the changes in PR #123"
```

**Authentication**: Requires GitHub Personal Access Token

---

## 📈 High Value - Install When Starting Analysis

### 4. ✅ SQLite MCP (Database Queries) - INSTALLED

**Status**: ✓ Connected
**What it does**: Query your ml_experiments.db database with natural language

**Installation**:
```bash
claude mcp add sqlite -- npx -y mcp-sqlite
```

**Use cases**:
- Complex SQL queries on your experiment database
- Cross-experiment comparisons
- Metrics analysis over time
- Model checkpoint queries
- Dataset lineage tracking

**Example queries**:
```
"Query ml_experiments.db to find the experiment with the best validation return"
"Show me all checkpoints from Experiment #10"
"Compare train vs validation returns for all experiments"
"Find all experiments that used learning rate 3e-4"
```

**Authentication**: None required!

---

### 5. ✅ Jupyter MCP (Interactive Notebooks) - INSTALLED

**Status**: ✓ Connected
**What it does**: Create and run Jupyter notebooks for analysis and visualization

**Installation**:
```bash
claude mcp add jupyter -- uvx jupyter-mcp-server@latest
```

**Use cases**:
- Interactive training curve visualization
- Backtest analysis notebooks
- Shareable experiment reports
- Data exploration
- Feature engineering experiments

**Example queries**:
```
"Create a Jupyter notebook that plots all training curves"
"Make a notebook analyzing the best experiment's episode returns"
"Generate a shareable report for Experiment #10"
"Create interactive plots of portfolio values over time"
```

**Authentication**: None required!

---

## 🔮 Future Installation - When Ready to Scale/Deploy

### 6. PostgreSQL MCP (Database Scaling)

**What it does**: When your SQLite database gets too large, migrate to PostgreSQL

**Installation**:
```bash
# You'll need PostgreSQL connection details
claude mcp add postgres \
  --env POSTGRES_CONNECTION_STRING=postgresql://user:pass@host:5432/dbname \
  -- npx -y @modelcontextprotocol/server-postgres
```

**Use cases**:
- Handle 1000s of experiments
- Concurrent experiment tracking
- Production-grade ML tracking
- Team collaboration on experiments
- Advanced analytics

**When to install**: When you have >50 experiments or need team access

**Authentication**: Requires PostgreSQL database credentials

---

### 7. Freqtrade MCP (Live Trading Bot)

**What it does**: Connect your trained RL agent to Freqtrade for paper/live trading

**Installation**:
```bash
# Requires Freqtrade installed and running
claude mcp add freqtrade \
  --env FREQTRADE_API_URL=http://localhost:8080 \
  -- npx -y @kukapay/freqtrade-mcp
```

**Use cases**:
- Paper trading with your trained agent
- Live deployment preparation
- Risk management integration
- Portfolio tracking
- Strategy backtesting

**Example queries**:
```
"Connect to Freqtrade and start paper trading with my best model"
"What's the current portfolio status?"
"Show recent trades and their performance"
"Set up a new trading strategy with model episode_1000.pth"
```

**When to install**: When your model is performing well and you want to test live

**Authentication**: Requires Freqtrade API credentials

---

### 8. Alpha Vantage MCP (Broader Financial Data)

**What it does**: Access stocks, forex, crypto, technical indicators, economic data

**Installation**:
```bash
# First, get free API key from: https://www.alphavantage.co/support/#api-key

claude mcp add alphavantage \
  --env ALPHAVANTAGE_API_KEY=your_key_here \
  -- npx -y @modelcontextprotocol/server-alphavantage
```

**Use cases**:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Economic calendar data
- Multi-asset correlations
- Fundamental data
- Alternative data sources

**Example queries**:
```
"Get RSI and MACD for Bitcoin"
"Show me correlation between Bitcoin and S&P 500"
"What economic events are happening this week?"
"Calculate Bollinger Bands for BTC-USD"
```

**When to install**: When you want to enhance features with technical indicators

**Authentication**: Requires free API key from alphavantage.co

---

## 🧪 Research & Development

### 9. ✅ Hugging Face MCP (ML Research) - INSTALLED + PYTHON INTEGRATION

**What it does**: Browse ML models, datasets, and research papers

**Status**: ✓ Connected (v0.2.32) + Python wrapper available

**Installation**:
```bash
# Correct package (tested and working!)
claude mcp add huggingface -- cmd /c npx -y @llmindset/hf-mcp-server
```

**Features**:
- 11 out of 13 tools enabled
- Access to Hugging Face models, datasets, and papers
- STDIO transport with API polling
- Internal API client for user configs
- **NEW**: Python integration via `tools/huggingface_mcp.py`

**Key Resources Found**:
- **CryptoTrader-LM**: BTC/ETH trading model (2022-2024 data)
- **Stock-trading-rl-agent**: PPO implementation
- **Financial-news-multisource**: 47M+ rows dataset
- **CryptoBERT**: Sentiment analysis models
- **Papers**: Meta-RL-Crypto, FLAG-Trader, Fin-R1

**Use cases**:
- Find state-of-art RL models ✅
- Access RL benchmark datasets ✅
- Research latest RL papers ✅
- Compare architectures ✅
- Find pre-trained models ✅
- **NEW**: Programmatic Python access ✅
- **NEW**: Sentiment analysis integration ✅
- **NEW**: Backtest enrichment ✅

**Example queries (Claude Code)**:
```
"Find latest RL papers on financial trading"
"Show me PPO implementations on Hugging Face"
"Search for Bitcoin trading datasets"
"What's the state-of-art in crypto RL trading?"
```

**Example usage (Python)**:
```python
from tools.huggingface_mcp import HuggingFaceClient, HFTradingResearch

# Search models
client = HuggingFaceClient()
models = client.search_models("bitcoin trading PPO", limit=5)

# Research assistant
research = HFTradingResearch()
report = research.generate_research_report()
print(report["recommendations"])

# Add sentiment to backtests
from tools.huggingface_mcp import HFDataEnricher
enricher = HFDataEnricher()
enriched_context = enricher.enrich_market_context(context, news_items)
```

**Documentation**:
- `HUGGINGFACE_MCP.md` - Complete integration guide
- `HUGGINGFACE_RL_RESOURCES.md` - Research findings
- `RESEARCH_PAPERS_HUGGINGFACE.md` - Paper summaries
- `tools/huggingface_mcp.py` - Python integration code
- `download_cryptotrader_lm.py` - Model download example

**Policy**: Registered in `policy/tools.yaml` (Class A: read-only, Class B: download)

**Authentication**: Optional (Hugging Face account for more access)

---

## 🛠️ Troubleshooting MCPs

### Check MCP Status
```bash
# List all installed MCPs
claude mcp list

# Get details on specific MCP
claude mcp get ccxt

# Remove a broken MCP
claude mcp remove mcp-name
```

### Common Issues

**MCP shows "Failed to connect"**:
1. Check if the npm package exists: `npx -y package-name --version`
2. Try reinstalling: `claude mcp remove name && claude mcp add name -- command`
3. Check authentication if required

**MCP not showing tools**:
1. Restart Claude Code
2. Check `/mcp` command to see available servers
3. Some MCPs take a few seconds to initialize

**Environment variables not working**:
1. Make sure no spaces around `=` in `--env KEY=VALUE`
2. Put values with spaces in quotes: `--env KEY="value with spaces"`
3. Check the MCP's documentation for exact variable names

---

## 📚 Quick Reference

### View Available MCPs
In Claude Code, type: `/mcp`

### Authenticate Remote MCPs
In Claude Code, type: `/mcp` and follow authentication flow

### MCP Resources
Use `@` to reference MCP resources:
```
@ccxt:ticker://BTC-USD
@github:issue://123
```

### Current Installation Status

✅ **All Installed & Working** (7/9 from recommended list):
- ccxt (Cryptocurrency data)
- sqlite (Database queries)
- pandas (Data analysis with pandas)
- github (Version control & PRs)
- jupyter (Interactive notebooks)
- playwright (Browser automation + Python wrapper)
- huggingface (ML research + **Python integration tool**)

---

## 🎯 Recommended Installation Sequence

For your Bitcoin RL trading project, install in this order:

**Week 1** (Immediate value):
1. ✅ CCXT - Already done!
2. ✅ SQLite - Already done!
3. ✅ Pandas - Already done!

**Week 2** (Collaboration):
4. ✅ GitHub - Already done!
5. ✅ Jupyter - Already done!

**Week 3** (Scaling):
6. PostgreSQL - When >50 experiments
7. Alpha Vantage - Add technical indicators

**Week 4** (Deployment):
8. Freqtrade - Paper trading
9. ✅ Hugging Face - Already done!

---

## 💡 Pro Tips

1. **Start Simple**: Install one MCP at a time, test it, then add more
2. **No Auth First**: Install MCPs without authentication first (CCXT, SQLite, Pandas)
3. **Test Before Production**: Use `/mcp` to verify connection before using in tasks
4. **Scope Matters**: Use `--scope user` for personal MCPs, `--scope project` for team-shared
5. **Clean Up**: Remove failed MCPs with `claude mcp remove name`

---

## 🔗 Additional Resources

- **Official MCP List**: https://github.com/modelcontextprotocol/servers
- **MCP Documentation**: https://modelcontextprotocol.io
- **Claude MCP Guide**: https://docs.claude.com/en/docs/claude-code/mcp
- **Your ML Tracking**: See `ML_TRACKING_INTEGRATION_COMPLETE.md`
- **Training Guide**: See `RUN_TRAINING.md`

---

**Generated**: October 20, 2025
**Project**: ultrathink-pilot
**Current Training**: Experiment #10 (PID 726) - 1000 episodes

**Next Steps**:
1. Install SQLite MCP for database queries
2. Install Pandas MCP for experiment analysis
3. Install GitHub MCP for version control
4. Test each one before moving to the next!
