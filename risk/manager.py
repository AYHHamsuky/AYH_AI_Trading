"""
Risk manager — position sizing, SL/TP, drawdown guard, kill switch.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
import numpy as np
from config import RISK_CONFIG as RC

logger = logging.getLogger(__name__)


@dataclass
class TradeOrder:
    symbol:     str
    direction:  int         # 0=SELL, 2=BUY
    lot_size:   float
    entry_price: float
    stop_loss:  float
    take_profit: float
    rr_ratio:   float
    risk_amount: float       # in account currency
    comment:    str = ""
    magic:      int = 202401


class RiskManager:
    """
    Stateful risk manager. Tracks open trades, daily PnL, drawdown.
    Call approve_trade() before placing any order.
    """

    def __init__(self, account_balance: float = 10_000):
        self._balance        = account_balance
        self._peak_balance   = account_balance
        self._daily_loss     = 0.0
        self._daily_reset_date = date.today()
        self._open_trades: dict[str, list] = {}    # symbol → list of orders
        self._killed         = False
        self._kill_reason    = ""   # "daily_loss" | "drawdown" | ""

    # ── Main entry point ──────────────────────────────────────────────────
    def approve_trade(
        self,
        symbol:     str,
        direction:  int,
        entry:      float,
        atr:        float,
        confidence: float,
        signal_meta: dict = None,
    ) -> Optional[TradeOrder]:
        """
        Returns a TradeOrder if the trade passes all risk checks, else None.
        """
        self._reset_daily_if_needed()

        if self._killed:
            logger.warning("KILL SWITCH ACTIVE — no new trades")
            return None

        # ── Kill switch checks ────────────────────────────────────────────
        if self._check_kill_switch():
            return None

        # ── Open trade limits ─────────────────────────────────────────────
        total_open  = sum(len(v) for v in self._open_trades.values())
        symbol_open = len(self._open_trades.get(symbol, []))

        if total_open >= RC["max_open_trades"]:
            logger.debug("Max open trades reached (%d)", total_open)
            return None
        if symbol_open >= RC["max_symbol_trades"]:
            logger.debug("Max trades for %s reached (%d)", symbol, symbol_open)
            return None

        # ── SL/TP calculation ─────────────────────────────────────────────
        sl_dist = atr * RC["sl_atr_mult"]
        tp_dist = atr * RC["tp_atr_mult"]
        if direction == 2:   # BUY
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:                # SELL
            sl = entry + sl_dist
            tp = entry - tp_dist
        rr = tp_dist / (sl_dist + 1e-9)

        if rr < RC["min_rr_ratio"]:
            logger.debug("RR %.2f below minimum %.2f — skipped", rr, RC["min_rr_ratio"])
            return None

        # ── Position sizing ───────────────────────────────────────────────
        lot_size = self._calculate_lot(
            symbol, entry, sl_dist, confidence
        )
        if lot_size <= 0:
            return None

        risk_amount = self._balance * RC["account_risk_pct"] / 100

        order = TradeOrder(
            symbol      = symbol,
            direction   = direction,
            lot_size    = round(lot_size, 2),
            entry_price = entry,
            stop_loss   = round(sl, 5),
            take_profit = round(tp, 5),
            rr_ratio    = round(rr, 2),
            risk_amount = round(risk_amount, 2),
            comment     = f"AI signal conf={confidence:.2f}",
        )
        logger.info(
            "ORDER APPROVED: %s %s lot=%.2f entry=%.5f sl=%.5f tp=%.5f RR=%.2f",
            ["SELL","?","BUY"][direction], symbol,
            lot_size, entry, sl, tp, rr,
        )
        return order

    # ── Position sizing ───────────────────────────────────────────────────
    def _calculate_lot(
        self, symbol: str, entry: float, sl_dist: float, confidence: float
    ) -> float:
        method = RC["position_sizing"]
        if method == "fixed_lot":
            return RC["fixed_lot"]

        risk_pct = RC["account_risk_pct"] / 100
        if method == "kelly":
            # Fractional Kelly: f* = edge/odds
            # We approximate win rate & avg win/loss from config
            kelly_fraction = RC["kelly_fraction"]
            # Scale by confidence
            risk_pct = risk_pct * kelly_fraction * confidence

        # Risk per lot ≈ sl_dist × contract_size (simplified for forex)
        contract_size = self._contract_size(symbol)
        risk_per_lot  = sl_dist * contract_size
        if risk_per_lot <= 0:
            return 0.0
        lot = (self._balance * risk_pct) / risk_per_lot
        lot = max(0.01, min(lot, 10.0))   # clamp 0.01 – 10 lots
        return round(lot, 2)

    @staticmethod
    def _contract_size(symbol: str) -> float:
        if "JPY" in symbol:
            return 1_000
        if "XAU" in symbol:
            return 100
        if "BTC" in symbol or "ETH" in symbol:
            return 1
        return 100_000   # standard forex lot

    # ── Kill switch ───────────────────────────────────────────────────────
    def _check_kill_switch(self) -> bool:
        daily_loss_pct = abs(self._daily_loss) / self._balance * 100
        drawdown_pct   = (self._peak_balance - self._balance) / self._peak_balance * 100

        if daily_loss_pct >= RC["max_daily_loss_pct"]:
            logger.critical("KILL SWITCH: daily loss %.2f%% ≥ limit %.2f%%",
                            daily_loss_pct, RC["max_daily_loss_pct"])
            self._killed = True
            self._kill_reason = "daily_loss"
            return True
        if drawdown_pct >= RC["max_drawdown_pct"]:
            logger.critical("KILL SWITCH: drawdown %.2f%% ≥ limit %.2f%%",
                            drawdown_pct, RC["max_drawdown_pct"])
            self._killed = True
            self._kill_reason = "drawdown"
            return True
        return False

    # ── Trade lifecycle ───────────────────────────────────────────────────
    def register_open(self, order: TradeOrder) -> None:
        self._open_trades.setdefault(order.symbol, []).append(order)

    def register_close(self, symbol: str, pnl: float, order: "Optional[TradeOrder]" = None) -> None:
        trades = self._open_trades.get(symbol, [])
        if trades:
            if order is not None and order in trades:
                trades.remove(order)
            else:
                trades.pop(0)
        self._balance    += pnl
        self._daily_loss += min(0.0, pnl)
        self._peak_balance = max(self._peak_balance, self._balance)
        logger.info("Trade closed: %s PnL=%.2f | Balance=%.2f", symbol, pnl, self._balance)
        # Check kill switch after each close
        self._check_kill_switch()

    # ── Trailing stop ─────────────────────────────────────────────────────
    def trailing_sl(
        self, order: TradeOrder, current_price: float, atr: float
    ) -> Optional[float]:
        """Return new SL if trailing stop should be updated, else None."""
        if not RC["trailing_stop"]:
            return None
        trail_dist = atr * RC["trailing_atr_mult"]
        if order.direction == 2:   # BUY
            new_sl = current_price - trail_dist
            if new_sl > order.stop_loss:
                return round(new_sl, 5)
        else:                      # SELL
            new_sl = current_price + trail_dist
            if new_sl < order.stop_loss:
                return round(new_sl, 5)
        return None

    # ── Portfolio stats ───────────────────────────────────────────────────
    def stats(self) -> dict:
        drawdown = (self._peak_balance - self._balance) / self._peak_balance * 100
        return {
            "balance":      round(self._balance, 2),
            "peak_balance": round(self._peak_balance, 2),
            "drawdown_pct": round(drawdown, 2),
            "daily_loss":   round(self._daily_loss, 2),
            "open_trades":  sum(len(v) for v in self._open_trades.values()),
            "kill_switch":  self._killed,
        }

    def reset_kill_switch(self) -> None:
        """Manual override — use with caution."""
        logger.warning("Kill switch reset by operator")
        self._killed = False
        self._kill_reason = ""

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._daily_reset_date:
            self._daily_loss      = 0.0
            self._daily_reset_date = today
            # Only reset kill switch if it was triggered by daily loss;
            # drawdown is cumulative and must be cleared manually.
            if self._kill_reason == "daily_loss":
                self._killed = False
                self._kill_reason = ""
