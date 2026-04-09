"""
MT5 live execution — place orders, manage positions, update trailing stops.
"""

import logging
import time
from datetime import datetime
from typing import Optional, List, Dict
from risk.manager import TradeOrder
from config import MT5_CONFIG

logger = logging.getLogger(__name__)


class MT5Broker:
    """
    Thin wrapper around MetaTrader5 Python API.
    Falls back gracefully if MT5 is unavailable (e.g. on cloud server
    without MT5 installed — use broker REST API instead in that case).
    """

    def __init__(self):
        self._connected = False
        self._positions: Dict[int, dict] = {}   # ticket → info

    # ── Connection ────────────────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
            return False

        if not self._mt5.initialize(
            path     = MT5_CONFIG["path"],
            login    = MT5_CONFIG["login"],
            password = MT5_CONFIG["password"],
            server   = MT5_CONFIG["server"],
        ):
            logger.error("MT5 connection failed: %s", self._mt5.last_error())
            return False

        info = self._mt5.account_info()
        logger.info(
            "MT5 connected | Account: %d | Balance: %.2f %s",
            info.login, info.balance, info.currency
        )
        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def account_balance(self) -> float:
        if not self._connected:
            return 0.0
        return self._mt5.account_info().balance

    # ── Order placement ───────────────────────────────────────────────────
    def place_order(self, order: TradeOrder) -> Optional[int]:
        """Place a market order. Returns ticket number or None on failure."""
        if not self._connected:
            logger.error("Not connected to MT5")
            return None

        mt5 = self._mt5
        direction = mt5.ORDER_TYPE_BUY if order.direction == 2 else mt5.ORDER_TYPE_SELL
        symbol_info = mt5.symbol_info(order.symbol)
        if symbol_info is None:
            logger.error("Symbol %s not found in MT5", order.symbol)
            return None

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       order.symbol,
            "volume":       float(order.lot_size),
            "type":         direction,
            "price":        mt5.symbol_info_tick(order.symbol).ask if order.direction == 2
                            else mt5.symbol_info_tick(order.symbol).bid,
            "sl":           float(order.stop_loss),
            "tp":           float(order.take_profit),
            "deviation":    MT5_CONFIG["deviation"],
            "magic":        MT5_CONFIG["magic"],
            "comment":      order.comment[:31],   # MT5 limit
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "Order failed for %s: retcode=%d, comment=%s",
                order.symbol, result.retcode, result.comment
            )
            return None

        ticket = result.order
        self._positions[ticket] = {
            "symbol":     order.symbol,
            "direction":  order.direction,
            "lot_size":   order.lot_size,
            "entry":      result.price,
            "sl":         order.stop_loss,
            "tp":         order.take_profit,
            "open_time":  datetime.utcnow(),
        }
        logger.info(
            "ORDER PLACED: ticket=%d %s %s lot=%.2f entry=%.5f SL=%.5f TP=%.5f",
            ticket, ["SELL","?","BUY"][order.direction], order.symbol,
            order.lot_size, result.price, order.stop_loss, order.take_profit
        )
        return ticket

    # ── Modify / close ────────────────────────────────────────────────────
    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Update stop loss for an existing position."""
        if not self._connected:
            return False
        mt5 = self._mt5
        pos  = mt5.positions_get(ticket=ticket)
        if not pos:
            logger.warning("Ticket %d not found for SL modification", ticket)
            return False
        p = pos[0]
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol":   p.symbol,
            "sl":       float(new_sl),
            "tp":       float(p.tp),
            "magic":    MT5_CONFIG["magic"],
        }
        result = mt5.order_send(request)
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            logger.info("SL updated: ticket=%d new_sl=%.5f", ticket, new_sl)
            if ticket in self._positions:
                self._positions[ticket]["sl"] = new_sl
        else:
            logger.error("SL update failed: retcode=%d", result.retcode)
        return ok

    def close_position(self, ticket: int) -> Optional[float]:
        """Close position by ticket. Returns realised PnL."""
        if not self._connected:
            return None
        mt5 = self._mt5
        pos  = mt5.positions_get(ticket=ticket)
        if not pos:
            return None
        p = pos[0]
        close_type  = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = mt5.symbol_info_tick(p.symbol).bid if close_type == mt5.ORDER_TYPE_SELL \
                      else mt5.symbol_info_tick(p.symbol).ask
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "position":     ticket,
            "symbol":       p.symbol,
            "volume":       p.volume,
            "type":         close_type,
            "price":        close_price,
            "deviation":    MT5_CONFIG["deviation"],
            "magic":        MT5_CONFIG["magic"],
            "comment":      "AI bot close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            pnl = p.profit
            logger.info("Position closed: ticket=%d PnL=%.2f", ticket, pnl)
            self._positions.pop(ticket, None)
            return pnl
        logger.error("Close failed: ticket=%d retcode=%d", ticket, result.retcode)
        return None

    def close_all(self) -> float:
        """Emergency close all positions. Returns total PnL."""
        total = 0.0
        for ticket in list(self._positions.keys()):
            pnl = self.close_position(ticket)
            if pnl is not None:
                total += pnl
        return total

    # ── Portfolio status ──────────────────────────────────────────────────
    def get_open_positions(self) -> List[dict]:
        if not self._connected:
            return []
        mt5 = self._mt5
        positions = mt5.positions_get(magic=MT5_CONFIG["magic"])
        if positions is None:
            return []
        return [
            {
                "ticket":    p.ticket,
                "symbol":    p.symbol,
                "type":      "BUY" if p.type == 0 else "SELL",
                "volume":    p.volume,
                "price_open": p.price_open,
                "sl":        p.sl,
                "tp":        p.tp,
                "profit":    p.profit,
                "time":      datetime.fromtimestamp(p.time),
            }
            for p in positions
        ]

    def get_current_price(self, symbol: str) -> Optional[float]:
        if not self._connected:
            return None
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return (tick.bid + tick.ask) / 2
