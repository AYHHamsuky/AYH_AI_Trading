"""
Backtesting engine — event-driven, walk-forward, with full metrics reporting.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from config import BACKTEST_CONFIG as BC, FEATURE_CONFIG as FC
from features.engineer import FeatureEngineer
from models.xgboost_model import XGBoostSignalModel
from models.lstm_model import LSTMForecaster
from models.ensemble import EnsembleSignalEngine
from risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol:      str
    direction:   int
    entry_time:  datetime
    entry_price: float
    stop_loss:   float
    take_profit: float
    lot_size:    float
    exit_time:   Optional[datetime]  = None
    exit_price:  Optional[float]     = None
    pnl:         float = 0.0
    exit_reason: str  = ""
    confidence:  float = 0.0


class BacktestEngine:
    """
    Event-driven backtester with walk-forward windows.
    Usage:
        engine = BacktestEngine("EURUSD", "H1")
        results = engine.run(df_ohlcv)
        engine.print_report()
    """

    def __init__(self, symbol: str, timeframe: str):
        self.symbol    = symbol
        self.timeframe = timeframe
        self._trades: List[BacktestTrade] = []
        self._equity_curve: List[float]   = []
        self._balance = BC["initial_balance"]

    # ── Main runner ───────────────────────────────────────────────────────
    def run(self, df_raw: pd.DataFrame, n_windows: int = None) -> dict:
        """Run walk-forward backtest. Returns metrics dict."""
        n_windows = n_windows or BC["walk_forward_windows"]
        engineer  = FeatureEngineer()
        df_feat   = engineer.build(df_raw)

        window_size = len(df_feat) // (n_windows + 1)
        window_results = []

        for w in range(n_windows):
            train_start = 0
            train_end   = int(window_size * (w + 1))
            test_start  = train_end
            test_end    = min(train_end + window_size, len(df_feat))
            if test_start >= len(df_feat):
                break

            df_train = df_feat.iloc[train_start:train_end]
            df_test  = df_feat.iloc[test_start:test_end]

            logger.info(
                "Window %d/%d — train: %d bars, test: %d bars",
                w+1, n_windows, len(df_train), len(df_test)
            )

            # Train models on window
            xgb_model  = XGBoostSignalModel(self.symbol, self.timeframe)
            lstm_model = LSTMForecaster(self.symbol, self.timeframe)
            try:
                xgb_model.train(df_train, verbose=False)
            except Exception as exc:
                logger.warning("XGBoost train failed (window %d): %s", w+1, exc)
                xgb_model = None
            try:
                lstm_model.train(df_train, verbose=0)
            except Exception as exc:
                logger.warning("LSTM train failed (window %d): %s", w+1, exc)
                lstm_model = None

            ensemble = EnsembleSignalEngine()
            risk_mgr  = RiskManager(self._balance)
            window_trades = self._simulate(df_test, df_raw.iloc[test_start:test_end],
                                           xgb_model, lstm_model, ensemble, risk_mgr)
            self._trades.extend(window_trades)
            window_results.append(self._window_metrics(window_trades))

        return self._aggregate_metrics(window_results)

    # ── Simulation ────────────────────────────────────────────────────────
    def _simulate(
        self, df_feat, df_ohlcv, xgb_model, lstm_model, ensemble, risk_mgr
    ) -> List[BacktestTrade]:
        trades: List[BacktestTrade] = []
        open_trade: Optional[BacktestTrade] = None

        commission = BC["commission_pct"]
        slippage   = BC["slippage_pct"]

        for i in range(60, len(df_feat)):
            bar_feat = df_feat.iloc[:i]
            bar      = df_ohlcv.iloc[i]
            price    = float(bar["close"])
            atr      = float(bar_feat.iloc[-1].get("atr", price * 0.001))

            # ── Check open trade exit ─────────────────────────────────────
            if open_trade is not None:
                exit_price, exit_reason = self._check_exit(open_trade, bar)
                if exit_price:
                    pnl = self._calc_pnl(open_trade, exit_price) * (1 - commission * 2)
                    open_trade.exit_time   = pd.to_datetime(df_feat.index[i])
                    open_trade.exit_price  = exit_price
                    open_trade.pnl         = pnl
                    open_trade.exit_reason = exit_reason
                    self._balance += pnl
                    self._equity_curve.append(self._balance)
                    risk_mgr.register_close(open_trade.symbol, pnl)
                    trades.append(open_trade)
                    open_trade = None

            # ── Generate new signal if no open trade ──────────────────────
            if open_trade is None:
                signal = ensemble.evaluate(
                    self.symbol, self.timeframe, bar_feat,
                    xgb_model=xgb_model, lstm_model=lstm_model,
                )
                if signal.is_actionable:
                    entry = price * (1 + slippage * (1 if signal.direction==2 else -1))
                    order = risk_mgr.approve_trade(
                        self.symbol, signal.direction, entry, atr, signal.confidence
                    )
                    if order:
                        open_trade = BacktestTrade(
                            symbol      = self.symbol,
                            direction   = signal.direction,
                            entry_time  = signal.timestamp,
                            entry_price = entry,
                            stop_loss   = order.stop_loss,
                            take_profit = order.take_profit,
                            lot_size    = order.lot_size,
                            confidence  = signal.confidence,
                        )
        return trades

    def _check_exit(
        self, trade: BacktestTrade, bar: pd.Series
    ):
        high = float(bar["high"])
        low  = float(bar["low"])
        if trade.direction == 2:   # BUY
            if low <= trade.stop_loss:
                return trade.stop_loss, "SL"
            if high >= trade.take_profit:
                return trade.take_profit, "TP"
        else:                      # SELL
            if high >= trade.stop_loss:
                return trade.stop_loss, "SL"
            if low <= trade.take_profit:
                return trade.take_profit, "TP"
        return None, ""

    @staticmethod
    def _calc_pnl(trade: BacktestTrade, exit_price: float) -> float:
        diff = exit_price - trade.entry_price
        if trade.direction == 0:   # SELL
            diff = -diff
        # Simplified PnL — in real use, factor in pip value and lot size
        contract = 100_000 if "USD" in trade.symbol[:6] else 100
        return diff * contract * trade.lot_size

    # ── Metrics ───────────────────────────────────────────────────────────
    def _window_metrics(self, trades: List[BacktestTrade]) -> dict:
        if not trades:
            return {}
        pnls  = [t.pnl for t in trades]
        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(trades)
        profit_factor = abs(sum(wins)) / (abs(sum(losses)) + 1e-9)
        return {
            "trades":        len(trades),
            "win_rate":      win_rate,
            "profit_factor": profit_factor,
            "total_pnl":     sum(pnls),
            "avg_win":       np.mean(wins) if wins else 0,
            "avg_loss":      np.mean(losses) if losses else 0,
            "max_win":       max(wins) if wins else 0,
            "max_loss":      min(losses) if losses else 0,
        }

    def _aggregate_metrics(self, windows: List[dict]) -> dict:
        if not windows:
            return {}
        agg = {}
        for k in windows[0]:
            vals = [w[k] for w in windows if k in w]
            agg[k]  = round(float(np.mean(vals)), 4)
            agg[k + "_std"] = round(float(np.std(vals)), 4)

        # Sharpe ratio from equity curve
        if len(self._equity_curve) > 1:
            rets = np.diff(self._equity_curve) / (np.array(self._equity_curve[:-1]) + 1e-9)
            agg["sharpe"] = round(float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)), 3)
        agg["final_balance"] = round(self._balance, 2)
        return agg

    def print_report(self) -> None:
        closed = [t for t in self._trades if t.exit_price]
        if not closed:
            print("No closed trades.")
            return
        metrics = self._window_metrics(closed)
        print("\n" + "="*50)
        print(f"BACKTEST REPORT — {self.symbol} {self.timeframe}")
        print("="*50)
        print(f"Total trades :  {metrics['trades']}")
        print(f"Win rate     :  {metrics['win_rate']:.1%}")
        print(f"Profit factor:  {metrics['profit_factor']:.2f}")
        print(f"Total PnL    :  ${metrics['total_pnl']:,.2f}")
        print(f"Avg win      :  ${metrics['avg_win']:,.2f}")
        print(f"Avg loss     :  ${metrics['avg_loss']:,.2f}")
        print(f"Final balance:  ${self._balance:,.2f}")
        print("="*50)
        # Auto-save results for dashboard
        self.save_results(metrics)

    def save_results(self, metrics: dict = None) -> str:
        """Save backtest results + trade list to JSON for the dashboard."""
        closed = [t for t in self._trades if t.exit_price]
        if metrics is None:
            metrics = self._window_metrics(closed) if closed else {}

        trades_data = []
        for t in closed:
            trades_data.append({
                "symbol":      t.symbol,
                "direction":   "BUY" if t.direction == 2 else "SELL",
                "entry_time":  str(t.entry_time),
                "entry_price": round(t.entry_price, 5),
                "exit_time":   str(t.exit_time),
                "exit_price":  round(t.exit_price, 5),
                "stop_loss":   round(t.stop_loss, 5),
                "take_profit": round(t.take_profit, 5),
                "lot_size":    t.lot_size,
                "pnl":         round(t.pnl, 2),
                "exit_reason": t.exit_reason,
                "confidence":  round(t.confidence, 3),
            })

        result = {
            "symbol":        self.symbol,
            "timeframe":     self.timeframe,
            "run_time":      datetime.now().isoformat(),
            "initial_balance": BC["initial_balance"],
            "final_balance": round(self._balance, 2),
            "metrics":       metrics,
            "equity_curve":  [round(e, 2) for e in self._equity_curve],
            "trades":        trades_data,
        }

        os.makedirs("logs", exist_ok=True)
        path = f"logs/backtest_{self.symbol}_{self.timeframe}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Backtest results saved → %s", path)
        return path
