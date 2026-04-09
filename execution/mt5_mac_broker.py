"""
MT5 Broker — Mac / Linux compatible via mt5linux socket bridge
──────────────────────────────────────────────────────────────
How it works on Mac:
  1. MT5 terminal runs natively (MetaQuotes Mac app) or via CrossOver/Wine
  2. mt5linux connects to it over a local socket — same API as Windows MetaTrader5
  3. Python code is 100% identical; the bridge handles the OS difference

Install:
  pip install mt5linux

Setup (one-time):
  - Open MT5 terminal on Mac
  - In MT5: Tools → Options → Expert Advisors → tick "Allow automated trading"
  - Run the bridge server (see README below)

Bridge server (run once in a separate terminal):
  python -c "import mt5linux; mt5linux.start_server()"
  # Default port 18812 — keep this terminal open while trading

Reference: https://github.com/lucas-campagna/mt5linux
"""

import logging
import time
from datetime import datetime
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class MT5MacBroker:
    """
    Full MT5 broker that works on Mac via mt5linux socket bridge.
    API is identical to the Windows MetaTrader5 package.
    """

    # MT5 timeframe constants (mirrored so we don't need mt5 imported at module level)
    TF = {
        "M1":1,"M2":2,"M3":3,"M4":4,"M5":5,"M6":6,"M10":10,"M12":12,
        "M15":15,"M20":20,"M30":30,"H1":16385,"H2":16386,"H3":16387,
        "H4":16388,"H6":16390,"H8":16392,"H12":16396,"D1":16408,
        "W1":32769,"MN1":49153,
    }

    def __init__(
        self,
        login:    int   = 0,
        password: str   = "",
        server:   str   = "",
        host:     str   = "127.0.0.1",  # mt5linux bridge host
        port:     int   = 18812,         # mt5linux bridge port
        magic:    int   = 202401,
        deviation:int   = 20,
    ):
        self.login    = login
        self.password = password
        self.server   = server
        self.host     = host
        self.port     = port
        self.magic    = magic
        self.deviation= deviation
        self._mt5     = None
        self._connected = False

    # ── Connection ────────────────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Connect to MT5 via mt5linux bridge.
        Make sure the bridge server is running first:
            python -c "import mt5linux; mt5linux.start_server()"
        """
        try:
            from mt5linux import MetaTrader5 as MT5
        except ImportError:
            logger.error(
                "mt5linux not installed.\n"
                "Run:  pip install mt5linux\n"
                "Then: python -c \"import mt5linux; mt5linux.start_server()\""
            )
            return False

        self._mt5 = MT5(host=self.host, port=self.port)

        # Initialize connection to the running MT5 terminal
        ok = self._mt5.initialize()
        if not ok:
            err = self._mt5.last_error()
            logger.error("MT5 initialize() failed: %s", err)
            logger.error(
                "Make sure:\n"
                "  1. MT5 terminal is open on your Mac\n"
                "  2. Bridge server is running: python -c \"import mt5linux; mt5linux.start_server()\"\n"
                "  3. Tools → Options → Expert Advisors → Allow automated trading is ticked"
            )
            return False

        # Login if credentials provided
        if self.login and self.password and self.server:
            ok = self._mt5.login(self.login, self.password, self.server)
            if not ok:
                logger.error("MT5 login failed: %s", self._mt5.last_error())
                return False

        info = self._mt5.account_info()
        if info is None:
            logger.error("Cannot retrieve account info — check credentials")
            return False

        logger.info(
            "MT5 connected (Mac bridge) | Account: %d | Balance: %.2f %s | Server: %s",
            info.login, info.balance, info.currency, info.server
        )
        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    # ── Account ───────────────────────────────────────────────────────────
    def account_balance(self) -> float:
        if not self._connected: return 0.0
        info = self._mt5.account_info()
        return float(info.balance) if info else 0.0

    def account_info(self) -> dict:
        if not self._connected: return {}
        info = self._mt5.account_info()
        if not info: return {}
        return {
            "login":          info.login,
            "balance":        float(info.balance),
            "equity":         float(info.equity),
            "margin":         float(info.margin),
            "free_margin":    float(info.margin_free),
            "unrealized_pnl": float(info.profit),
            "currency":       info.currency,
            "leverage":       info.leverage,
            "server":         info.server,
            "broker":         info.company,
        }

    # ── Pricing ───────────────────────────────────────────────────────────
    def get_price(self, symbol: str) -> Optional[Dict]:
        if not self._connected: return None
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None: return None
        return {
            "bid":    tick.bid,
            "ask":    tick.ask,
            "mid":    (tick.bid + tick.ask) / 2,
            "spread": tick.ask - tick.bid,
            "time":   datetime.fromtimestamp(tick.time),
        }

    def get_current_price(self, symbol: str) -> Optional[float]:
        p = self.get_price(symbol)
        return p["mid"] if p else None

    # ── OHLCV data (for the live loop) ────────────────────────────────────
    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 500):
        """Fetch OHLCV bars directly from MT5 — no yfinance needed in live mode."""
        import pandas as pd
        if not self._connected: return pd.DataFrame()
        tf   = self.TF.get(timeframe, self.TF["H1"])
        bars = self._mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if bars is None or len(bars) == 0:
            logger.warning("No bars returned for %s %s", symbol, timeframe)
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"time":"datetime","tick_volume":"volume"})
        df = df.set_index("datetime")[["open","high","low","close","volume"]]
        return df.sort_index()

    # ── Order placement ───────────────────────────────────────────────────
    def place_market_order(
        self,
        symbol:    str,
        direction: int,      # 2=BUY, 0=SELL
        lot_size:  float,
        sl:        float,
        tp:        float,
        comment:   str = "AI bot",
    ) -> Optional[int]:
        """Place a market order. Returns ticket number or None on failure."""
        if not self._connected:
            logger.error("Not connected to MT5")
            return None

        mt5 = self._mt5
        order_type = mt5.ORDER_TYPE_BUY if direction == 2 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("Cannot get tick for %s — is the symbol in Market Watch?", symbol)
            return None
        price = tick.ask if direction == 2 else tick.bid

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(round(lot_size, 2)),
            "type":         order_type,
            "price":        price,
            "sl":           float(round(sl, 5)),
            "tp":           float(round(tp, 5)),
            "deviation":    self.deviation,
            "magic":        self.magic,
            "comment":      comment[:31],
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        for attempt in range(3):
            result = mt5.order_send(request)
            if result is None:
                logger.warning("order_send returned None (attempt %d)", attempt+1)
                time.sleep(self.deviation / 1000)
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                ticket = result.order
                logger.info(
                    "✅ ORDER FILLED | %s %s | lot=%.2f | price=%.5f | SL=%.5f | TP=%.5f | ticket=%d",
                    "BUY" if direction==2 else "SELL", symbol,
                    lot_size, result.price, sl, tp, ticket
                )
                return ticket
            else:
                logger.warning(
                    "Order attempt %d failed: retcode=%d (%s)",
                    attempt+1, result.retcode, result.comment
                )
                if attempt < 2:
                    time.sleep(2)

        logger.error("All order attempts failed for %s", symbol)
        return None

    # ── Alias for AutoTrader compatibility ────────────────────────────────
    def place_order(self, order) -> Optional[int]:
        """Accept a TradeOrder dataclass (from risk manager)."""
        return self.place_market_order(
            symbol    = order.symbol,
            direction = order.direction,
            lot_size  = order.lot_size,
            sl        = order.stop_loss,
            tp        = order.take_profit,
            comment   = order.comment[:31] if order.comment else "AI bot",
        )

    @staticmethod
    def lot_to_units(symbol: str, lot_size: float) -> float:
        """MT5 uses lot sizes directly — no conversion needed."""
        return lot_size

    # ── Modify / close ────────────────────────────────────────────────────
    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify stop loss of an open position."""
        if not self._connected: return False
        mt5 = self._mt5
        pos = mt5.positions_get(ticket=ticket)
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
            "magic":    self.magic,
        }
        result = mt5.order_send(request)
        ok = result and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            logger.info("SL updated: ticket=%d new_sl=%.5f", ticket, new_sl)
        else:
            logger.error("SL update failed: %s", result.comment if result else "None")
        return ok

    # Alias for auto_trader compatibility
    def modify_trade_sl(self, trade_id, new_sl: float) -> bool:
        try:
            return self.modify_sl(int(trade_id), new_sl)
        except (ValueError, TypeError):
            return False

    def close_position(self, ticket: int) -> Optional[float]:
        """Close a position by ticket number."""
        if not self._connected: return None
        mt5 = self._mt5
        pos = mt5.positions_get(ticket=ticket)
        if not pos: return None
        p = pos[0]
        close_type  = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
        close_price = mt5.symbol_info_tick(p.symbol).bid \
                      if close_type == mt5.ORDER_TYPE_SELL \
                      else mt5.symbol_info_tick(p.symbol).ask
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "position":     ticket,
            "symbol":       p.symbol,
            "volume":       p.volume,
            "type":         close_type,
            "price":        close_price,
            "deviation":    self.deviation,
            "magic":        self.magic,
            "comment":      "AI bot close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            pnl = p.profit
            logger.info("Position closed: ticket=%d PnL=%.2f", ticket, pnl)
            return pnl
        logger.error("Close failed: ticket=%d | %s", ticket,
                     result.comment if result else "None")
        return None

    # Alias for auto_trader
    def close_trade(self, trade_id) -> Optional[float]:
        try:
            return self.close_position(int(trade_id))
        except (ValueError, TypeError):
            return None

    def close_all(self) -> float:
        """Emergency close all open positions (magic number only)."""
        total = 0.0
        if not self._connected: return total
        positions = self._mt5.positions_get(magic=self.magic)
        if positions:
            for p in positions:
                pnl = self.close_position(p.ticket)
                if pnl is not None:
                    total += pnl
        logger.warning("EMERGENCY CLOSE ALL | Total PnL: %.2f", total)
        return total

    # ── Open positions ────────────────────────────────────────────────────
    def get_open_trades(self) -> List[Dict]:
        if not self._connected: return []
        positions = self._mt5.positions_get(magic=self.magic)
        if not positions: return []
        return [
            {
                "id":           str(p.ticket),
                "ticket":       p.ticket,
                "symbol":       p.symbol,
                "direction":    "BUY" if p.type == 0 else "SELL",
                "units":        p.volume,
                "lot_size":     p.volume,
                "entry_price":  p.price_open,
                "sl":           p.sl,
                "tp":           p.tp,
                "unrealized_pnl": p.profit,
                "open_time":    datetime.fromtimestamp(p.time),
            }
            for p in positions
        ]

    def get_open_positions(self) -> List[Dict]:
        return self.get_open_trades()

    def get_closed_trades(self, count: int = 50) -> List[Dict]:
        if not self._connected: return []
        from datetime import timedelta
        history = self._mt5.history_deals_get(
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        )
        if not history: return []
        deals = []
        for d in history[-count:]:
            deals.append({
                "id":         str(d.order),
                "symbol":     d.symbol,
                "direction":  "BUY" if d.type == 0 else "SELL",
                "pnl":        d.profit,
                "volume":     d.volume,
                "price":      d.price,
                "open_time":  datetime.fromtimestamp(d.time),
            })
        return deals
