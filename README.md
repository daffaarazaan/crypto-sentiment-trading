
# Crypto Price + Sentiment Trading Dashboard

📊 Interactive dashboard built with **Streamlit** to test algorithmic trading strategies on **BTC & ETH**, combining:
- Technical indicators (RSI, EMA spreads, volatility)
- News sentiment (FinBERT / CardiffNLP)

## 🚀 Features
- Train **Price-only** vs **Price+Sentiment** ML models (XGBoost)
- Backtest with realistic trading costs
- Equity curves for BTC & ETH
- Combined metrics table (Sharpe, Annual Return, MaxDD, Accuracy)
- Upload your own news dataset or use Kaggle's crypto-news

## ⚙️ Tech Stack
- Python, Streamlit
- yFinance for price data
- HuggingFace Transformers (FinBERT) for sentiment
- XGBoost for ML signals
- TA-Lib indicators via `ta` package

## 📈 Live Demo
👉 [Try it here](https://crypto-sentiment-trading.streamlit.app) (Streamlit Cloud)

## 🛠 How to Run Locally
```bash
git clone https://github.com/<your-username>/crypto-sentiment-trading.git
cd crypto-sentiment-trading
pip install -r requirements.txt
streamlit run app.py

