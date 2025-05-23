# dca-bot-with-LLM-sentiment-analysis


![newplot](https://github.com/user-attachments/assets/3fb6ec51-cd30-4268-9571-c924c65b4545)
The most noticable part of the LLM behavior is that it is able to have a peak performance at the end 2024 and start of 2025. This is not meant to be deployed yet, but to showcase that LLM is able to be used to automate decision-making, in which we limitted our choices to robust and well-tested companies that outperformed the regular market, which is the Magnificent 7.

# Mag7SentimentBot 📈🧠

A weekly **Dollar-Cost Averaging (DCA)** trading bot powered by **LLM-based sentiment analysis** to trade the **Magnificent 7 stocks** (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA). This strategy uses news sentiment to decide which stocks to buy or sell and runs on the **Lumibot** trading framework with **Ollama** as a local language model interface.

---

## 🔍 Strategy Overview

This bot implements a hybrid **indexing + sentiment-based DCA strategy**:

- **Week 1**: Allocate **50% of the portfolio** based on **market capitalization weights** (similar to SPY or QQQ).
- **Subsequent Weeks**:
  - **Buy** stocks with **positive sentiment ≥ 0.25** using ~2.5% of the portfolio (or available cash).
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
