# Gemini Repository Review Guide

This guide explains how to use Google's Gemini AI to perform comprehensive code reviews of the UltraThink Pilot repository.

## Quick Start

### 1. Get Your Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key" or "Get API Key"
4. Copy your API key

### 2. Configure Environment

Edit the `.env` file in the project root:

```bash
GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE
```

Replace `YOUR_ACTUAL_API_KEY_HERE` with the API key you copied.

### 3. Run the Review

```bash
# Basic review (analyzes current directory)
python3 gemini_repo_review.py

# Specify custom output file
python3 gemini_repo_review.py --output my_review.md

# Use different Gemini model
python3 gemini_repo_review.py --model gemini-pro

# Review different repository
python3 gemini_repo_review.py --repo-path /path/to/other/repo
```

## What Gets Analyzed

The script performs comprehensive analysis of:

### Code Components
- ✅ **Agents** - MR-SR, ERS agent implementations
- ✅ **Backtesting** - Data fetching, portfolio simulation, metrics
- ✅ **RL System** - Trading environment, PPO agent, training
- ✅ **ML Persistence** - Experiment tracking, model registry
- ✅ **Orchestration** - Agent pipeline coordination
- ✅ **Tests** - Unit and integration tests
- ✅ **Configuration** - YAML/JSON policy files
- ✅ **Documentation** - README, guides, comments
- ✅ **Scripts** - Shell scripts, utilities

### Review Aspects

1. **Architecture & Design**
   - System architecture and component interactions
   - Design patterns and best practices
   - Modularity and separation of concerns
   - Scalability considerations

2. **Code Quality**
   - Code organization and structure
   - Naming conventions
   - Code duplication
   - Error handling

3. **Trading System Specific**
   - Agent coordination
   - Risk management
   - Backtesting robustness
   - RL implementation

4. **Testing & Reliability**
   - Test coverage
   - Integration testing
   - Mock design
   - Error scenarios

5. **Performance**
   - Algorithm efficiency
   - Resource utilization
   - Bottlenecks
   - GPU optimization

6. **Security**
   - API key handling
   - Input validation
   - Dependency security

7. **Documentation**
   - Code documentation
   - Setup instructions
   - Configuration guides

## Output Format

The script generates a comprehensive markdown report with:

- **Repository Structure** - File categorization and overview
- **Detailed Analysis** - Per-component code review
- **Strengths** - What's implemented well
- **Issues** - Problems and concerns
- **Recommendations** - Specific improvements
- **Priority Ratings** - High/Medium/Low priorities

## Available Models

### Recommended Models

```bash
# Best for comprehensive analysis (default)
--model gemini-2.0-flash-thinking-exp-01-21

# Fast analysis
--model gemini-2.0-flash-exp

# Most capable (higher cost)
--model gemini-1.5-pro
```

### Model Comparison

| Model | Speed | Depth | Cost | Best For |
|-------|-------|-------|------|----------|
| `gemini-2.0-flash-thinking-exp-01-21` | Medium | Deep | Low | Comprehensive reviews |
| `gemini-2.0-flash-exp` | Fast | Good | Low | Quick analysis |
| `gemini-1.5-pro` | Slow | Excellent | Higher | Critical audits |

## Command Line Options

```bash
python3 gemini_repo_review.py [OPTIONS]

Options:
  --repo-path PATH    Path to repository (default: current directory)
  --output FILE       Output markdown file (default: auto-generated)
  --model MODEL       Gemini model name (default: gemini-2.0-flash-thinking-exp-01-21)
  --api-key KEY       Gemini API key (overrides .env)
  -h, --help          Show help message
```

## Examples

### Example 1: Basic Review
```bash
python3 gemini_repo_review.py
```
Output: `gemini_review_20250123_143022.md`

### Example 2: Custom Output
```bash
python3 gemini_repo_review.py --output my_comprehensive_review.md
```

### Example 3: Different Model
```bash
python3 gemini_repo_review.py --model gemini-1.5-pro --output deep_analysis.md
```

### Example 4: API Key from Command Line
```bash
python3 gemini_repo_review.py --api-key AIza... --output review.md
```

## Understanding the Output

### Sample Output Structure

```markdown
# Gemini Repository Review - UltraThink Pilot

**Generated:** 2025-01-23T14:30:22
**Model:** gemini-2.0-flash-thinking-exp-01-21
**Files Analyzed:** 45
**Total Lines:** 5,234

---

# Repository Structure Overview
...

---

# Comprehensive Review

## 1. Architecture & Design

### ✅ Strengths
- Clear separation between agents, backtesting, and RL
- Well-defined communication protocols
- Process isolation for agents

### ⚠️ Issues
- Potential circular dependency in orchestration
- Missing graceful degradation for network failures

### 💡 Recommendations
1. Implement circuit breaker pattern (Priority: High)
2. Add health check endpoints (Priority: Medium)
...
```

## Troubleshooting

### Error: "Gemini API key not found"

**Solution:** Set your API key in `.env`:
```bash
GEMINI_API_KEY=AIzaSyC...your_key_here
```

### Error: "Quota exceeded"

**Solution:** Gemini free tier has limits. Either:
- Wait for quota reset (usually daily)
- Upgrade to paid tier
- Use smaller model: `--model gemini-2.0-flash-exp`

### Error: "API call failed"

**Possible causes:**
- Invalid API key
- Network connectivity issues
- API service outage

**Solutions:**
1. Verify API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Check internet connection
3. Try again later

### Script hangs during analysis

**Solution:** Large repositories may take time. The script shows progress:
```
🔄 Calling Gemini API (call #1)...
```

For very large repos, consider reviewing specific components separately.

## Advanced Usage

### Programmatic Usage

```python
from gemini_repo_review import GeminiRepoReviewer

# Initialize reviewer
reviewer = GeminiRepoReviewer(api_key="your_key", model="gemini-2.0-flash-thinking-exp-01-21")

# Scan repository
files = reviewer.scan_repository(".")

# Generate summary
summary = reviewer.create_code_summary(files)

# Get review
review = reviewer.review_with_gemini(summary, review_type="comprehensive")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'model': reviewer.model_name,
    'statistics': reviewer.stats,
    'gemini_review': review
}
reviewer.save_review(results, "output.md")
```

### Custom Review Types

```python
# Architecture-focused review
review = reviewer.review_with_gemini(content, review_type="architecture")

# Security audit
review = reviewer.review_with_gemini(content, review_type="security")

# Performance analysis
review = reviewer.review_with_gemini(content, review_type="performance")
```

## Cost Considerations

### Free Tier Limits (as of 2025)
- 60 requests per minute
- 1,500 requests per day
- 1 million tokens per minute

### Tips to Reduce Costs
1. Use `gemini-2.0-flash-exp` for quick reviews
2. Review specific directories instead of entire repo
3. Limit files per category (script already does this)
4. Cache results and review incrementally

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Gemini Code Review

on:
  pull_request:
    branches: [main]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install google-generativeai python-dotenv

      - name: Run Gemini Review
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: |
          python3 gemini_repo_review.py --output review.md

      - name: Upload Review
        uses: actions/upload-artifact@v3
        with:
          name: gemini-review
          path: review.md
```

## Best Practices

1. **Version Control** - Don't commit `.env` with real API keys
2. **Regular Reviews** - Run reviews before major releases
3. **Incremental Analysis** - Review changed files between commits
4. **Multiple Perspectives** - Use different review types for different focuses
5. **Act on Feedback** - Prioritize High-priority recommendations

## Getting Help

- **Gemini API Docs**: https://ai.google.dev/docs
- **API Key Issues**: https://aistudio.google.com/app/apikey
- **Script Issues**: Check this README or contact maintainers

## Changelog

### Version 1.0 (2025-01-23)
- Initial release
- Comprehensive repository scanning
- Multi-category file analysis
- Structured Gemini prompting
- Markdown report generation
- Support for multiple Gemini models

---

**Ready to get started?**

```bash
# 1. Get your API key from https://aistudio.google.com/app/apikey
# 2. Add to .env: GEMINI_API_KEY=your_key_here
# 3. Run: python3 gemini_repo_review.py
```

Happy reviewing! 🚀
