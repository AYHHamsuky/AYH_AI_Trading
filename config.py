"""
AI Forex Trading System — Configuration
Author: AYH HAMSUKY ENTERPRISES
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

# ─────────────────────────────────────────────
# Trading pairs & timeframes
# ─────────────────────────────────────────────
SYMBOLS = {
    "forex":     ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"],
    "metals":    ["XAUUSD"],
    "crypto":    ["BTCUSD", "ETHUSD"],
    # Deriv synthetic indices — 24/7, no news events, perfect for automation
    "synthetic": ["V75", "V50", "V25", "BOOM1000", "CRASH1000"],
}

TIMEFRAMES = {
    "scalp":    ["M1", "M5", "M15"],
    "intraday": ["H1", "H4"],
    "swing":    ["D1"],
}

# Primary signal timeframe → confirmation timeframe
TIMEFRAME_MAP = {
    "M5":  "M15",
    "M15": "H1",
    "H1":  "H4",
    "H4":  "D1",
}

# ─────────────────────────────────────────────
# Deriv API configuration
# ─────────────────────────────────────────────
DERIV_CONFIG = {
    # Account modes
    "demo_token":  os.getenv("DERIV_API_TOKEN_DEMO", ""),
    "live_token":  os.getenv("DERIV_API_TOKEN_LIVE", ""),
    "active_mode": os.getenv("DERIV_MODE", "demo"),   # "demo" | "live"

    # Trade execution settings for Deriv contracts
    "contract_duration":  5,        # minutes (for binary/vanilla)
    "multiplier":         100,      # CFD multiplier (1, 10, 50, 100, 200, 500, 1000)
    # Note: higher multiplier = higher leverage = higher risk
    # Recommended: 100 for forex, 50 for gold, 10 for crypto

    # Stake sizing (account currency)
    "min_stake":          1.0,      # minimum $1 per trade
    "max_stake":          500.0,    # maximum $500 per trade

    # WebSocket
    "ws_url":      "wss://ws.binaryws.com/websockets/v3?app_id=1089",
    "app_id":      1089,
}

# ─────────────────────────────────────────────
# Data sources
# ─────────────────────────────────────────────
DATA_CONFIG = {
    "primary":      "deriv",       # deriv | yfinance | ccxt
    "fallback":     "yfinance",    # used when Deriv unavailable
    "lookback_bars": 2000,
    "live_bars":    500,
}

# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────
FEATURE_CONFIG = {
    "rsi_period":      14,
    "macd_fast":       12,
    "macd_slow":       26,
    "macd_signal":     9,
    "bb_period":       20,
    "bb_std":          2.0,
    "atr_period":      14,
    "ema_periods":     [9, 21, 50, 200],
    "lag_periods":     [1, 2, 3, 5, 10],
    "return_periods":  [1, 5, 10, 20],
    "volume_periods":  [10, 20],
    "target_horizon":  5,           # bars ahead for label generation
    "target_pct":      0.003,       # 0.3 % move to label BUY/SELL
}

# ─────────────────────────────────────────────
# XGBoost model
# ─────────────────────────────────────────────
XGBOOST_CONFIG = {
    "n_estimators":    500,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":           0.1,
    "reg_alpha":       0.01,
    "reg_lambda":      1.0,
    "use_gpu":         False,       # True if CUDA available on VPS
    "early_stopping":  50,
    "eval_metric":     "mlogloss",
    "classes":         ["SELL", "HOLD", "BUY"],   # 0, 1, 2
    "retrain_every":   500,         # bars between full retrains
}

# ─────────────────────────────────────────────
# LSTM model
# ─────────────────────────────────────────────
LSTM_CONFIG = {
    "sequence_len":    60,          # look-back window
    "hidden_units":    [128, 64],   # two LSTM layers
    "dropout":         0.2,
    "dense_units":     32,
    "epochs":          100,
    "batch_size":      32,
    "patience":        15,          # early stopping patience
    "learning_rate":   0.001,
    "forecast_steps":  5,           # bars ahead
    "scale_features":  True,
    "retrain_every":   1000,
}

# ─────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────
ENSEMBLE_CONFIG = {
    "xgb_weight":          0.60,    # weight for XGBoost signal
    "lstm_weight":         0.40,    # weight for LSTM signal
    "min_confidence":      0.58,    # minimum probability to trade
    "mtf_confirm":         True,    # require higher-TF confirmation
    "mtf_weight":          0.30,    # higher-TF override weight
    "session_filter":      False,   # disabled — V75 synthetic trades 24/7
    "sessions": {                   # UTC hours
        "london":   (7, 16),
        "newyork":  (12, 21),
        "tokyo":    (0, 8),
    },
}

# ─────────────────────────────────────────────
# Risk management
# ─────────────────────────────────────────────
RISK_CONFIG = {
    "account_risk_pct":    1.0,     # % of balance per trade
    "max_open_trades":     5,
    "max_symbol_trades":   2,
    "max_daily_loss_pct":  5.0,     # kill switch (relaxed for synthetics)
    "max_drawdown_pct":    12.0,    # kill switch (relaxed for walk-forward)
    "sl_atr_mult":         1.5,     # SL = 1.5 × ATR
    "tp_atr_mult":         3.0,     # TP = 3.0 × ATR (1:2 RR)
    "trailing_stop":       True,
    "trailing_atr_mult":   1.0,
    "min_rr_ratio":        1.5,     # discard trades below this RR
    "position_sizing":     "fixed_lot", # fixed_lot for predictable risk
    "fixed_lot":           0.01,
    "kelly_fraction":      0.25,    # fractional Kelly
}

# ─────────────────────────────────────────────
# Backtesting
# ─────────────────────────────────────────────
BACKTEST_CONFIG = {
    "initial_balance":     10_000,
    "commission_pct":      0.0002,  # 2 pips round-trip
    "slippage_pct":        0.0001,
    "spread_pips":         2,
    "use_tick_data":       False,
    "train_ratio":         0.70,
    "val_ratio":           0.15,
    "test_ratio":          0.15,
    "walk_forward_windows": 5,
}

# ─────────────────────────────────────────────
# MT5 connection
# ─────────────────────────────────────────────
MT5_CONFIG = {
    "login":    int(os.getenv("MT5_LOGIN", "0") if str(os.getenv("MT5_LOGIN", "0")).lstrip("-").isdigit() else "0"),
    "password": os.getenv("MT5_PASSWORD", ""),
    "server":   os.getenv("MT5_SERVER", ""),
    "path":     os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe"),
    "magic":    202401,             # unique EA magic number
    "deviation": 20,               # max slippage in points
}

# ─────────────────────────────────────────────
# Telegram alerts
# ─────────────────────────────────────────────
TELEGRAM_CONFIG = {
    "token":   os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    "alerts": {
        "trade_open":    True,
        "trade_close":   True,
        "signal_gen":    True,
        "daily_report":  True,
        "kill_switch":   True,
        "error":         True,
    },
}

# ─────────────────────────────────────────────
# Auto-execution engine
# ─────────────────────────────────────────────
AUTO_TRADER_CONFIG = {
    "enabled":             False,        # master on/off switch
    "mode":                "paper",      # "paper" | "live"  — always start paper!
    "broker":              "oanda",      # "mt5" | "oanda" | "mt5_bridge"
    # Signal quality gates (ALL must pass before executing)
    "min_ensemble_conf":   0.68,         # XGBoost + LSTM ensemble minimum
    "min_smc_conf":        0.60,         # SMC engine minimum
    "require_smc_confirm": True,         # SMC direction must agree with ensemble
    "require_mtf_confirm": True,         # higher-TF must agree
    "signal_persistence":  2,            # signal must hold for N bars before execution
    "cooldown_bars":       10,           # min bars between trades on same symbol
    # Session & news filters
    "trade_london":        True,
    "trade_newyork":       True,
    "trade_tokyo":         False,
    "avoid_news_minutes":  30,           # skip trading N min before/after high-impact news
    # Execution
    "max_trades_per_day":  6,
    "max_trades_per_symbol": 2,
    "retry_attempts":      3,
    "retry_delay_sec":     2,
    # Paper trading
    "paper_initial_balance": 10_000,
    "paper_log_file":      "logs/paper_trades.json",
}

# ─────────────────────────────────────────────
# OANDA REST API (Mac-native broker)
# ─────────────────────────────────────────────
OANDA_CONFIG = {
    "api_key":      os.getenv("OANDA_API_KEY", ""),
    "account_id":   os.getenv("OANDA_ACCOUNT_ID", ""),
    "environment":  os.getenv("OANDA_ENV", "practice"),   # "practice" | "live"
    "base_url_practice": "https://api-fxpractice.oanda.com/v3",
    "base_url_live":     "https://api-fxtrade.oanda.com/v3",
    # OANDA instrument mapping
    "instruments": {
        "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD",
        "USDJPY": "USD_JPY", "AUDUSD": "AUD_USD",
        "USDCHF": "USD_CHF", "XAUUSD": "XAU_USD",
        "BTCUSD": "BTC_USD",
    },
}


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "saved_models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
DATA_DIR   = os.path.join(BASE_DIR, "data_cache")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)
