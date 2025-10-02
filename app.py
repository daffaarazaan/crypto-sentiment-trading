# app.py
# Streamlit dashboard: Crypto Price + Sentiment Trading (BTC & ETH)
# Author: <your name> — Portfolio Project
# Educational demo only, not financial advice.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ------------------------- UI -------------------------
st.set_page_config(page_title="Crypto Sentiment Trading", layout="wide")
st.title("Crypto Price + Sentiment Trading — BTC & ETH")

with st.sidebar:
    st.header("Settings")
    start = st.date_input("Start date", value=pd.to_datetime("2019-01-01"))
    end   = st.date_input("End date",   value=pd.to_datetime("today"))
    thr   = st.slider("Decision threshold (P>thr → long)", 0.50, 0.70, 0.55, 0.01)
    cost  = st.slider("Trading cost (bps per trade)", 0, 50, 10, 1)
    assets = st.multiselect("Assets", ["BTC-USD", "ETH-USD"], default=["BTC-USD","ETH-USD"])
    max_rows = st.number_input("Max news rows (for speed)", 1000, 50000, 8000, 1000)
    model_name = st.selectbox("Sentiment model", ["ProsusAI/finbert", "cardiffnlp/twitter-roberta-base-sentiment"])

st.caption("This app trains a price-only vs price+sentiment classifier, then backtests with trading costs.")

# ------------------------- Helper functions -------------------------
def backtest(prob, ret, thr=0.55, cost_bp=10):
    prob = np.asarray(prob); ret = np.asarray(ret)
    sig = (prob > thr).astype(int)
    trades = np.abs(np.diff(np.r_[0, sig]))
    cost = trades * (cost_bp/10000.0)
    strat_ret = sig * ret - cost
    equity = (1 + strat_ret).cumprod()
    return pd.DataFrame({"signal": sig, "strategy_ret": strat_ret, "equity": equity}), int(trades.sum())

def sharpe(returns, periods=252):
    r = np.asarray(returns)
    return float(np.sqrt(periods) * r.mean() / (r.std() + 1e-9))

def ann_return(equity, periods=252):
    e = np.asarray(equity); n = len(e)
    return np.nan if n==0 else (e[-1] ** (periods/n) - 1)

def max_dd(equity):
    e = pd.Series(np.asarray(equity))
    return float((e.cummax() - e).max())

def build_asset_pipeline(asset, sent_daily, start, end, split_date, thr, cost):
    # 1) Download prices
    px = yf.download(asset, start=str(start), end=str(end), interval="1d", auto_adjust=True).dropna()

    # Flatten MultiIndex columns if they exist
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = ["_".join([str(x) for x in tup if x != ""]).strip() for tup in px.columns]

    price_col = "Close" if "Close" in px.columns else ("Adj Close" if "Adj Close" in px.columns else None)
    assert price_col is not None, f"No Close/Adj Close in {asset} data."
    price = px[price_col].squeeze(); price.name = "price"

    # 2) Features
    px["ret"]   = price.pct_change()
    px["rsi14"] = RSIIndicator(price, 14).rsi()
    ema10       = EMAIndicator(price, 10).ema_indicator()
    ema20       = EMAIndicator(price, 20).ema_indicator()
    px["ema_spread"] = (ema10 - ema20) / price
    px["ret_lag1"]   = px["ret"].shift(1)
    px["vol_20"]     = px["ret"].rolling(20).std()

    # 3) Merge sentiment
    df = px.copy()
    df["date"] = pd.to_datetime(df.index).normalize()
    df = df.reset_index(drop=True)

    sd = sent_daily.copy()
    sd["date"] = pd.to_datetime(sd["date"]).dt.normalize()
    df = df.merge(sd[["date","sent_mean","sent_std","sent_n"]], on="date", how="left")

    for c in ["sent_mean","sent_std","sent_n"]:
        if c not in df.columns:
            df[c] = 0.0
    df[["sent_mean","sent_std","sent_n"]] = df[["sent_mean","sent_std","sent_n"]].fillna(0)

    # 4) Target
    df["target"] = (df["ret"].shift(-1) > 0).astype(int)

    feat_price = ["ret_lag1","rsi14","ema_spread","vol_20"]
    feat_all   = feat_price + ["sent_mean","sent_std","sent_n"]
    dfm = df.dropna(subset=feat_all + ["target"]).copy()

    # 5) Train/test split
    train = dfm[dfm["date"] < split_date]
    test  = dfm[dfm["date"] >= split_date]

    Xtr_p, ytr = train[feat_price], train["target"]
    Xte_p, yte = test[feat_price],  test["target"]

    Xtr_a, Xte_a = train[feat_all], test[feat_all]

    # 6) Train models
    m_base = XGBClassifier(n_estimators=300, max_depth=3, subsample=0.9, colsample_bytree=0.9, random_state=42)
    m_base.fit(Xtr_p, ytr)
    p_base = m_base.predict_proba(Xte_p)[:,1]

    m_sent = XGBClassifier(n_estimators=300, max_depth=3, subsample=0.9, colsample_bytree=0.9, random_state=42)
    m_sent.fit(Xtr_a, ytr)
    p_sent = m_sent.predict_proba(Xte_a)[:,1]

    # 7) Backtests
    bt_base, ntr_base = backtest(p_base, test["ret"].values, thr=thr, cost_bp=cost)
    bt_sent, ntr_sent = backtest(p_sent, test["ret"].values, thr=thr, cost_bp=cost)

    # 8) Metrics
    metrics = [
        {"Model":"Price-only", "Accuracy":round(accuracy_score(yte, (p_base>0.5).astype(int)),4),
         "Sharpe":round(sharpe(bt_base["strategy_ret"]),2),
         "AnnReturn":f"{ann_return(bt_base['equity']):.2%}",
         "MaxDD":f"{max_dd(bt_base['equity']):.2%}", "Trades":ntr_base},
        {"Model":"Price+Sent", "Accuracy":round(accuracy_score(yte, (p_sent>0.5).astype(int)),4),
         "Sharpe":round(sharpe(bt_sent["strategy_ret"]),2),
         "AnnReturn":f"{ann_return(bt_sent['equity']):.2%}",
         "MaxDD":f"{max_dd(bt_sent['equity']):.2%}", "Trades":ntr_sent},
        {"Model":"Buy&Hold", "Accuracy":"", "Sharpe":"",
         "AnnReturn":f"{ann_return((1+test['ret']).cumprod()):.2%}",
         "MaxDD":f"{max_dd((1+test['ret']).cumprod()):.2%}", "Trades":0},
    ]

    curves = {"Price-only": bt_base["equity"].values,
              "Price+Sent": bt_sent["equity"].values,
              "Buy&Hold":   (1+test["ret"]).cumprod()}
    dates  = test["date"].values

    return pd.DataFrame(metrics), curves, dates

# ------------------------- Sentiment pipeline -------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

@st.cache_resource
def load_sentiment_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=mdl, tokenizer=tok, truncation=True)

def score_sentiment(df_news, clf, batch=64):
    if df_news.empty: 
        return pd.DataFrame(columns=["date","sent_mean","sent_std","sent_n"])
    label_map = {"positive": 1, "negative": -1, "neutral": 0}
    scores = []
    texts = df_news["headline"].tolist()
    for i in range(0, len(texts), batch):
        outs = clf(texts[i:i+batch])
        scores.extend([label_map.get(o["label"].lower(), 0) for o in outs])
    df_news["sent"] = scores
    df_news["date"] = pd.to_datetime(df_news["published_at"]).dt.normalize()
    sent_daily = df_news.groupby("date")["sent"].agg(sent_mean="mean", sent_std="std", sent_n="count").reset_index()
    return sent_daily

# ------------------------- Load News -------------------------
def pick_col(cols, candidates):
    """Helper: find the first matching column name from a list of candidates"""
    cols_low = [c.lower() for c in cols]
    for c in candidates:
        if c.lower() in cols_low:
            return cols[cols_low.index(c.lower())]
    return None

uploaded = st.file_uploader("Upload a news CSV with a datetime + headline column (optional).")
if uploaded:
    df_news = pd.read_csv(uploaded)
else:
    import kagglehub, glob, os
    path = kagglehub.dataset_download("oliviervha/crypto-news")
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    df_news = pd.read_csv(csvs[0])

# Try to detect date and headline columns
cdate = pick_col(df_news.columns.tolist(), ["published_at","date","datetime","time","created_at","timestamp","pub_date"])
ctext = pick_col(df_news.columns.tolist(), ["headline","title","text","news","content","summary"])

if not cdate or not ctext:
    st.error("Could not find a valid datetime and headline/text column in the news dataset.")
    st.stop()

# Rename for consistency
df_news = df_news[[cdate, ctext]].rename(columns={cdate:"published_at", ctext:"headline"})
df_news["published_at"] = pd.to_datetime(df_news["published_at"], errors="coerce")
df_news = df_news.dropna(subset=["published_at","headline"]).head(int(max_rows))

# ------------------------- Run for assets -------------------------
split_date = pd.Timestamp("2024-01-01")

all_metrics = []
for asset in assets:
    st.subheader(f"Results for {asset}")
    metrics, curves, dates = build_asset_pipeline(asset, sent_daily, start, end, split_date, thr, cost)
    st.dataframe(metrics, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(dates, curves["Price+Sent"], label="Price+Sentiment")
    ax.plot(dates, curves["Price-only"], label="Price-only")
    ax.plot(dates, curves["Buy&Hold"], label="Buy & Hold")
    ax.set_title(f"Equity Curve — {asset}")
    ax.legend()
    st.pyplot(fig)

    metrics["Asset"] = asset
    all_metrics.append(metrics)

if all_metrics:
    df_all = pd.concat(all_metrics, ignore_index=True)
    st.subheader("Combined BTC & ETH Metrics")
    st.dataframe(df_all, use_container_width=True)
