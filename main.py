"""
AYH AI Trading Bot — Main Orchestrator
────────────────────────────────────────
Supports:
  • Deriv API (demo + live) — primary broker
  • MT5 Mac bridge          — alternative broker
  • yfinance                — data fallback

Usage:
  # Backtest (uses yfinance/Deriv for historical data)
  python main.py --mode backtest --broker deriv --symbols EURUSD XAUUSD V75 --tf H1

  # Train models
  python main.py --mode train --broker deriv --symbols EURUSD XAUUSD V75 --tf H1

  # Live trading — paper mode (no real orders)
  python main.py --mode live --broker deriv --account demo --auto paper --symbols EURUSD V75 --tf H1

  # Live trading — real orders on DEMO account
  python main.py --mode live --broker deriv --account demo --auto live --symbols EURUSD V75 --tf H1

  # Live trading — real orders on LIVE account (be careful!)
  python main.py --mode live --broker deriv --account live --auto live --symbols EURUSD --tf H1
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from config import DATA_CONFIG, XGBOOST_CONFIG, TIMEFRAME_MAP, AUTO_TRADER_CONFIG, DERIV_CONFIG, SYMBOLS as SYMBOL_GROUPS
from data.fetcher import DataFetcher
from features.engineer import FeatureEngineer
from features.smc import SMCEngine
from models.xgboost_model import XGBoostSignalModel
from models.lstm_model import LSTMForecaster
from models.ensemble import EnsembleSignalEngine
from risk.manager import RiskManager
from backtest.engine import BacktestEngine
from execution.auto_trader import AutoTrader
from alerts.notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trading_bot.log"),
    ],
)
logger = logging.getLogger("main")


class TradingBot:

    def __init__(
        self,
        symbols:    list,
        timeframes: list,
        mode:       str = "live",
        broker:     str = "deriv",     # "deriv" | "mt5"
        account:    str = "demo",      # "demo" | "live"
        auto_mode:  str = "paper",     # "paper" | "live"
    ):
        self.symbols    = symbols
        self.timeframes = timeframes
        self.mode       = mode
        self.broker_name= broker
        self.account    = account
        self.auto_mode  = auto_mode
        self._running   = False

        self.engineer   = FeatureEngineer()
        self.smc_engine = SMCEngine(swing_lookback=8)
        self.ensemble   = EnsembleSignalEngine()
        self.notifier   = TelegramNotifier()

        self.xgb_models:  Dict[str, XGBoostSignalModel] = {}
        self.lstm_models: Dict[str, LSTMForecaster]     = {}
        self._bar_counts: Dict[str, int]                 = {}

        self.broker       = None
        self.fetcher      = None
        self.risk_mgr     = RiskManager(10_000)
        self.auto_trader  = None
        self.tick_streamer = None

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ── Setup ─────────────────────────────────────────────────────────────
    def _setup_broker(self) -> bool:
        if self.broker_name == "deriv":
            return self._setup_deriv()
        return self._setup_mt5()

    def _setup_deriv(self) -> bool:
        from execution.deriv_broker import DerivClient
        token_key = f"DERIV_API_TOKEN_{self.account.upper()}"
        token     = os.getenv(token_key, "")
        if not token:
            logger.error(
                "No Deriv token found.\n"
                "Get yours at: app.deriv.com/account/api-token\n"
                "Then add to .env:  %s=your_token", token_key
            )
            return False

        self.broker = DerivClient(token, account_type=self.account)
        if not self.broker.connect():
            return False

        balance = self.broker.account_balance()
        info    = self.broker.account_info()
        logger.info(
            "Deriv [%s] | Login: %s | Balance: %.2f %s | Virtual: %s",
            self.account.upper(), info.get("login_id"),
            balance, info.get("currency","USD"),
            info.get("is_virtual", True)
        )

        # Set up data fetcher using Deriv
        self.fetcher = DataFetcher(
            deriv_token = token,
            deriv_mode  = self.account,
        )
        return True

    def _setup_mt5(self) -> bool:
        from execution.mt5_mac_broker import MT5MacBroker
        self.broker = MT5MacBroker(
            login    = int(os.getenv("MT5_LOGIN", "0")),
            password = os.getenv("MT5_PASSWORD", ""),
            server   = os.getenv("MT5_SERVER", ""),
        )
        if not self.broker.connect():
            return False
        self.fetcher = DataFetcher(fallback="yfinance")
        return True

    # ── Entry point ───────────────────────────────────────────────────────
    def start(self) -> None:
        logger.info("=" * 60)
        logger.info("AYH AI Trading Bot v4")
        logger.info("Broker: %s [%s] | AutoExec: %s",
                    self.broker_name.upper(), self.account.upper(), self.auto_mode.upper())
        logger.info("Symbols: %s", self.symbols)
        logger.info("=" * 60)

        if self.mode == "backtest":
            self._run_backtest()
            return
        if self.mode == "train":
            self._run_training()
            return

        # Live mode — connect broker
        logger.info("Connecting to %s...", self.broker_name)
        if not self._setup_broker():
            logger.error("Broker connection failed")
            sys.exit(1)

        balance = self.broker.account_balance()
        self.risk_mgr = RiskManager(balance)

        # Auto-trader
        self.auto_trader = AutoTrader(
            broker       = self.broker,
            risk_manager = self.risk_mgr,
            notifier     = self.notifier,
            mode         = self.auto_mode,
        )
        self.auto_trader.enable()

        # Start live tick streamer for real-time prices
        if self.broker_name == "deriv":
            self._start_tick_streamer()

        self.notifier.info(
            f"🤖 AYH Bot started\n"
            f"Broker: {self.broker_name.upper()} [{self.account.upper()}]\n"
            f"Balance: ${balance:,.2f}\n"
            f"Auto: {'📄 Paper' if self.auto_mode=='paper' else '💰 LIVE'}\n"
            f"Symbols: {', '.join(self.symbols)}"
        )

        # Start subscriber polling so users can /subscribe to receive signals
        self.notifier.start_polling()

        # Give notifier access to auto_trader, risk_mgr, broker for /commands
        self.notifier.set_context(
            auto_trader=self.auto_trader,
            risk_manager=self.risk_mgr,
            broker=self.broker,
        )

        self._load_or_train_models()
        self._running = True
        self._live_loop()

    # ── Live tick streamer ─────────────────────────────────────────────────
    def _start_tick_streamer(self) -> None:
        try:
            from execution.deriv_broker import DerivTickStreamer
            token_key = f"DERIV_API_TOKEN_{self.account.upper()}"
            token     = os.getenv(token_key, "")
            self.tick_streamer = DerivTickStreamer(token, self.symbols)
            self.tick_streamer.start()
            logger.info("Real-time tick streamer started for %s", self.symbols)
        except Exception as exc:
            logger.warning("Tick streamer failed to start: %s", exc)

    # ── Live loop ─────────────────────────────────────────────────────────
    def _live_loop(self) -> None:
        logger.info("Live trading loop started — polling every 60s")
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                logger.exception("Tick error: %s", exc)
                self.notifier.error_alert(str(exc))
            time.sleep(60)

    def _tick(self) -> None:
        for symbol in self.symbols:
            for tf in self.timeframes:
                key = f"{symbol}_{tf}"
                try:
                    self._process(symbol, tf, key)
                except Exception as exc:
                    logger.warning("Processing error %s: %s", key, exc)

        if self.risk_mgr.stats()["kill_switch"] and self._running:
            logger.critical("Kill switch triggered — stopping")
            self._running = False

    def _process(self, symbol: str, tf: str, key: str) -> None:
        # ── Fetch OHLCV (Deriv API primary) ───────────────────────────────
        df_raw = self.fetcher.fetch(symbol, tf, bars=DATA_CONFIG["live_bars"])
        if df_raw.empty or len(df_raw) < 60:
            return

        # ── Feature engineering with SMC ─────────────────────────────────
        df_feat = self.engineer.build(df_raw, add_labels=False, add_smc=True)
        if len(df_feat) < 50:
            return

        # ── Higher-TF confirmation ────────────────────────────────────────
        htf    = TIMEFRAME_MAP.get(tf)
        df_htf = None
        if htf:
            try:
                raw_htf = self.fetcher.fetch(symbol, htf, bars=200)
                if not raw_htf.empty:
                    df_htf = self.engineer.build(raw_htf, add_labels=False, add_smc=False)
            except Exception:
                pass

        # ── Scheduled retraining ──────────────────────────────────────────
        self._bar_counts[key] = self._bar_counts.get(key, 0) + 1
        if self._bar_counts[key] % XGBOOST_CONFIG["retrain_every"] == 0:
            self._retrain(symbol, tf, df_raw)

        # ── Signal generation ─────────────────────────────────────────────
        xgb    = self.xgb_models.get(key)
        lstm   = self.lstm_models.get(key)
        signal = self.ensemble.evaluate(symbol, tf, df_feat, df_htf, xgb, lstm)
        smc_sig= self.smc_engine.get_smc_signal(df_raw)

        # ── Notify signal ─────────────────────────────────────────────────
        if signal.direction != 1:  # not HOLD — broadcast to all subscribers
            self.notifier.signal_generated(
                symbol      = symbol,
                timeframe   = tf,
                direction   = signal.direction_name,
                confidence  = signal.confidence,
                xgb_dir     = ["SELL", "HOLD", "BUY"][signal.xgb_signal],
                lstm_dir    = ["SELL", "HOLD", "BUY"][signal.lstm_signal],
                smc_signal  = smc_sig,
                entry_price = signal.close,
                atr         = signal.atr,
                df          = df_raw,
            )

        # ── Auto-execute ──────────────────────────────────────────────────
        if self.auto_trader and self.auto_trader.is_active:
            trade_log = self.auto_trader.on_signal(signal, smc_sig, df_feat)
            if trade_log:
                logger.info(
                    "TRADE EXECUTED: %s %s | conf=%.2f | mode=%s",
                    trade_log.direction, symbol,
                    trade_log.confidence, trade_log.mode
                )

    # ── Model management ──────────────────────────────────────────────────
    def _load_or_train_models(self) -> None:
        for symbol in self.symbols:
            for tf in self.timeframes:
                key  = f"{symbol}_{tf}"
                xgb  = XGBoostSignalModel(symbol, tf)
                lstm = LSTMForecaster(symbol, tf)
                if not xgb.load():
                    logger.info("Training models for %s ...", key)
                    try:
                        df_raw = self.fetcher.fetch(
                            symbol, tf, bars=DATA_CONFIG["lookback_bars"]
                        )
                        self._train_pair(symbol, tf, df_raw, xgb, lstm)
                    except Exception as exc:
                        logger.error("Train failed for %s: %s", key, exc)
                else:
                    lstm.load()
                    logger.info("Models loaded: %s", key)
                self.xgb_models[key]  = xgb
                self.lstm_models[key] = lstm

    def _train_pair(self, symbol, tf, df_raw, xgb, lstm) -> None:
        df_feat = self.engineer.build(df_raw, add_smc=True)
        try:
            m = xgb.train(df_feat)
            logger.info("XGBoost OK: %s_%s | cv_acc=%.3f", symbol, tf, m.get("cv_accuracy_mean",0))
        except Exception as exc:
            logger.error("XGBoost train failed %s_%s: %s", symbol, tf, exc)
        try:
            m = lstm.train(df_feat)
            logger.info("LSTM OK: %s_%s | val_loss=%.6f", symbol, tf, m.get("val_loss",0))
        except Exception as exc:
            logger.error("LSTM train failed %s_%s: %s", symbol, tf, exc)

    def _retrain(self, symbol: str, tf: str, df_raw) -> None:
        df_feat = self.engineer.build(df_raw, add_smc=True)
        key     = f"{symbol}_{tf}"
        if x := self.xgb_models.get(key):
            try: x.retrain(df_feat)
            except Exception as exc: logger.warning("Retrain XGB failed: %s", exc)
        if l := self.lstm_models.get(key):
            try: l.train(df_feat)
            except Exception as exc: logger.warning("Retrain LSTM failed: %s", exc)

    # ── Backtest ──────────────────────────────────────────────────────────
    def _run_backtest(self) -> None:
        fetcher = DataFetcher()   # uses token from .env or falls back to yfinance
        for symbol in self.symbols:
            for tf in self.timeframes:
                logger.info("Backtesting %s %s ...", symbol, tf)
                try:
                    df_raw = fetcher.fetch(symbol, tf, bars=DATA_CONFIG["lookback_bars"])
                    engine = BacktestEngine(symbol, tf)
                    results = engine.run(df_raw)
                    engine.print_report()
                except Exception as exc:
                    logger.error("Backtest failed %s %s: %s", symbol, tf, exc)

    def _run_training(self) -> None:
        fetcher = DataFetcher()
        for symbol in self.symbols:
            for tf in self.timeframes:
                key = f"{symbol}_{tf}"
                logger.info("Training %s ...", key)
                try:
                    df_raw = fetcher.fetch(symbol, tf, bars=DATA_CONFIG["lookback_bars"])
                    xgb    = XGBoostSignalModel(symbol, tf)
                    lstm   = LSTMForecaster(symbol, tf)
                    self._train_pair(symbol, tf, df_raw, xgb, lstm)
                except Exception as exc:
                    logger.error("Training failed %s: %s", key, exc)
        logger.info("Training complete — models in saved_models/")

    # ── Shutdown ──────────────────────────────────────────────────────────
    def _shutdown(self, *_) -> None:
        logger.info("Shutting down...")
        self._running = False
        self.notifier.stop_polling()
        if self.tick_streamer:
            self.tick_streamer.stop()
        if self.auto_trader:
            stats = self.auto_trader.stats()
            self.notifier.daily_report({
                "balance":      self.broker.account_balance() if self.broker else 0,
                "daily_pnl":    -self.risk_mgr.stats()["daily_loss"],
                "trades_today": stats["trades_today"],
                "win_rate":     stats["win_rate"],
                "drawdown_pct": self.risk_mgr.stats()["drawdown_pct"],
                "open_trades":  stats["open_trades"],
            })
        if self.broker:
            self.broker.disconnect()
        sys.exit(0)


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AYH AI Trading Bot")
    p.add_argument("--mode",    choices=["live","backtest","train"], default="backtest")
    p.add_argument("--broker",  choices=["deriv","mt5"],             default="deriv")
    p.add_argument("--account", choices=["demo","live"],             default="demo")
    p.add_argument("--auto",    choices=["paper","live"],            default="paper",
                   help="paper=log only  live=real orders")
    _ALL_SYMBOLS = [s for group in SYMBOL_GROUPS.values() for s in group]
    p.add_argument("--symbols", nargs="+", default=_ALL_SYMBOLS)
    p.add_argument("--tf",      nargs="+", default=["H1"])
    args = p.parse_args()

    bot = TradingBot(
        symbols    = args.symbols,
        timeframes = args.tf,
        mode       = args.mode,
        broker     = args.broker,
        account    = args.account,
        auto_mode  = args.auto,
    )
    bot.start()
