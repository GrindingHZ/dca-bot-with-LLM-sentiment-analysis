# dca-bot-with-LLM-sentiment-analysis

# Mag7SentimentBot 📈🧠

A weekly **Dollar-Cost Averaging (DCA)** trading bot powered by **LLM-based sentiment analysis** to trade the **Magnificent 7 stocks** (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA). This strategy uses news sentiment to decide which stocks to buy or sell and runs on the **Lumibot** trading framework with **Ollama** as a local language model interface.

---

## 🔍 Strategy Overview

This bot implements a hybrid **indexing + sentiment-based DCA strategy**:

- **Week 1**: Allocate **50% of the portfolio** based on **market capitalization weights** (similar to SPY or QQQ).
- **Subsequent Weeks**:
  - **Buy** stocks with **positive sentiment ≥ 0.5** using ~5% of the portfolio (or available cash).
  - **Sell** stocks with **negative sentiment ≤ -0.8** by liquidating their full position.
  - **Hold** if no significant sentiment signals are present.

---

## 🧠 How Sentiment Analysis Works

1. Fetch recent news (past 7 days) using `get_web_deets()` (via Serper API or custom scraper).
2. Send news summaries to a **local LLM** (Ollama with `qwen2.5:14b`) using a prompt template.
3. Parse the LLM's response (structured JSON) to extract:
   - `sentiment`: a label (e.g., `"positive"`, `"negative"`)
   - `score`: a float in the range [-1, 1]
4. Use the score to drive weekly buy/sell decisions.

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
