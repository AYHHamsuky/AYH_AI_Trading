# AYH AI Trading Bot — Mac + MT5 Setup Guide

**Stack:** Python · XGBoost · LSTM · Smart Money Concepts · MT5 (Mac) · Streamlit

---

## Prerequisites

- macOS (any recent version)
- MT5 terminal installed → https://www.metatrader5.com/en/terminal/mac
- Python 3.11+ → https://brew.sh then `brew install python@3.11`
- Your broker MT5 login (XM, FBS, HotForex, Alpari, etc.)

---

## Step 1 — Install dependencies

```bash
cd ai_trader
pip install -r requirements.txt
```

---

## Step 2 — Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:
```
MT5_LOGIN=123456789
MT5_PASSWORD=YourPassword
MT5_SERVER=XM-Real18          # find in MT5: Help → About
TELEGRAM_BOT_TOKEN=...        # optional
TELEGRAM_CHAT_ID=...          # optional
```

---

## Step 3 — Enable automated trading in MT5

1. Open MT5 on Mac
2. Menu: **Tools → Options → Expert Advisors**
3. Tick: ☑ **Allow automated trading**
4. Click OK

---

## Step 4 — Start the MT5 Mac bridge

Open a **dedicated terminal** and run this — keep it open while trading:

```bash
python -c "import mt5linux; mt5linux.start_server()"
# Output: Server started on 127.0.0.1:18812
```

This bridges Python to your running MT5 terminal via a local socket.

---

## Step 5 — Test the connection

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from execution.mt5_mac_broker import MT5MacBroker
import os
b = MT5MacBroker(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))
print('Connected:', b.connect())
print('Balance:  ', b.account_balance())
print('Info:     ', b.account_info())
"
```

---

## Step 6 — Run the dashboard

```bash
streamlit run dashboard.py
```

Opens at http://localhost:8501

---

## Step 7 — Train models (first time)

```bash
python main.py --mode train --symbols EURUSD GBPUSD XAUUSD --tf H1
# Models saved to saved_models/
```

---

## Step 8 — Backtest

```bash
python main.py --mode backtest --symbols EURUSD XAUUSD --tf H1
```

Only go live when win rate > 52% and profit factor > 1.3 consistently.

---

## Step 9 — Run the bot (PAPER mode first)

```bash
# Paper mode — real signals, no real orders
python main.py --mode live --auto paper --symbols EURUSD XAUUSD --tf H1

# Live mode — sends real orders to MT5
python main.py --mode live --auto live --symbols EURUSD XAUUSD --tf H1
```

---

## Auto-execution quality gates

Before any real order fires, ALL 6 gates must pass:

| Gate | Description |
|------|-------------|
| Ensemble confidence | XGBoost + LSTM combined ≥ 68% |
| SMC confidence | Smart Money signal ≥ 60% |
| SMC direction agrees | SMC and ML agree on direction |
| Not HOLD | Signal must be BUY or SELL |
| Active session | London (07-16 UTC) or NY (12-21 UTC) |
| Kill switch clear | Daily loss < 3%, drawdown < 8% |

---

## File structure

```
ai_trader/
├── main.py                        ← Bot entry point
├── dashboard.py                   ← Streamlit dashboard
├── config.py                      ← All settings
├── .env.example                   ← Credential template
├── data/fetcher.py                ← yfinance / CCXT data
├── features/
│   ├── engineer.py                ← 40+ indicators + SMC injection
│   └── smc.py                     ← Order blocks, sweeps, inducement, FVG
├── models/
│   ├── xgboost_model.py           ← BUY/HOLD/SELL classifier (+ SMC features)
│   ├── lstm_model.py              ← BiLSTM price forecaster
│   └── ensemble.py                ← Weighted signal fusion
├── risk/manager.py                ← Kelly sizing, kill switch, trailing SL
├── execution/
│   ├── mt5_mac_broker.py          ← MT5 Mac bridge broker ← USE THIS
│   └── auto_trader.py             ← Auto-execution engine (6 quality gates)
├── backtest/engine.py             ← Walk-forward backtester
└── alerts/notifier.py             ← Telegram alerts
```

---

## Troubleshooting

**"mt5linux not installed"**
```bash
pip install mt5linux
```

**"MT5 initialize() failed"**
- Is MT5 terminal open on your Mac?
- Is the bridge server running?
- Is "Allow automated trading" ticked in MT5 options?

**"No bars returned"**
- Is the symbol added to Market Watch in MT5?
- Right-click Market Watch → Show All

**Bridge server keeps stopping**
- Run it in a dedicated terminal window that you don't close
- Or use: `nohup python -c "import mt5linux; mt5linux.start_server()" &`
