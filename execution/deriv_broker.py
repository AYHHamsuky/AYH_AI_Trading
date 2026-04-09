"""
Deriv WebSocket API Client
──────────────────────────
Supports:
  • Real-time tick streaming
  • Full OHLCV candle history (for backtesting + live data)
  • Demo AND Live account switching via API token
  • Direct trade execution (buy/sell/close)
  • Account balance, open positions, trade history

Setup (2 minutes):
  1. Log in at app.deriv.com
  2. Go to: Account Settings → API Token
  3. Create token with: Read + Trade + Payments + Admin scopes
  4. Add to .env:  DERIV_API_TOKEN_DEMO=your_demo_token
                  DERIV_API_TOKEN_LIVE=your_live_token

Deriv WebSocket docs: https://api.deriv.com/
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Any

import pandas as pd
import numpy as np
import websocket   # websocket-client

logger = logging.getLogger(__name__)

# ── Deriv symbol mapping ───────────────────────────────────────────────────
DERIV_SYMBOLS = {
    # Forex
    "EURUSD":  "frxEURUSD",
    "GBPUSD":  "frxGBPUSD",
    "USDJPY":  "frxUSDJPY",
    "AUDUSD":  "frxAUDUSD",
    "USDCHF":  "frxUSDCHF",
    "USDCAD":  "frxUSDCAD",
    "NZDUSD":  "frxNZDUSD",
    "EURGBP":  "frxEURGBP",
    "EURJPY":  "frxEURJPY",
    "GBPJPY":  "frxGBPJPY",
    # Metals
    "XAUUSD":  "frxXAUUSD",   # Gold
    "XAGUSD":  "frxXAGUSD",   # Silver
    # Crypto (Deriv offers these)
    "BTCUSD":  "cryBTCUSD",
    "ETHUSD":  "cryETHUSD",
    "LTCUSD":  "cryLTCUSD",
    # Deriv Synthetic Indices (unique to Deriv — always open, 24/7)
    "V10":     "R_10",         # Volatility 10 Index
    "V25":     "R_25",         # Volatility 25 Index
    "V50":     "R_50",         # Volatility 50 Index
    "V75":     "R_75",         # Volatility 75 Index
    "V100":    "R_100",        # Volatility 100 Index
    "BOOM1000":"BOOM1000",     # Boom 1000
    "CRASH1000":"CRASH1000",  # Crash 1000
    "JUMP10":  "JD10",         # Jump 10 Index
    "JUMP25":  "JD25",         # Jump 25 Index
    "JUMP50":  "JD50",         # Jump 50 Index
}

# Reverse mapping
DERIV_TO_SYMBOL = {v: k for k, v in DERIV_SYMBOLS.items()}

# Granularity mapping (seconds)
GRANULARITY = {
    "M1":  60,    "M2":  120,  "M3":  180,  "M5":  300,
    "M10": 600,   "M15": 900,  "M30": 1800,
    "H1":  3600,  "H2":  7200, "H4":  14400, "H8":  28800,
    "D1":  86400, "W1":  604800,
}

DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"


class DerivClient:
    """
    Synchronous wrapper around Deriv WebSocket API.
    All calls block until response arrives (with timeout).
    """

    def __init__(self, api_token: str, account_type: str = "demo"):
        self.api_token    = api_token
        self.account_type = account_type   # "demo" | "live"
        self._ws:         Optional[websocket.WebSocket] = None
        self._connected   = False
        self._req_id      = 0
        self._timeout     = 20   # seconds

    # ── Connection ────────────────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            self._ws = websocket.create_connection(
                DERIV_WS_URL,
                timeout=self._timeout,
            )
            # Authorize
            resp = self._call({"authorize": self.api_token})
            if "error" in resp:
                logger.error("Deriv auth failed: %s", resp["error"]["message"])
                return False

            acct = resp.get("authorize", {})
            bal  = acct.get("balance", 0)
            curr = acct.get("currency", "")
            logger.info(
                "Deriv connected [%s] | Account: %s | Balance: %s %.2f",
                self.account_type.upper(), acct.get("email", ""),
                curr, bal
            )
            self._connected = True
            return True
        except Exception as exc:
            logger.error("Deriv connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        if self._ws:
            try: self._ws.close()
            except: pass
        self._connected = False

    def reconnect(self) -> bool:
        self.disconnect()
        time.sleep(2)
        return self.connect()

    # ── Account ───────────────────────────────────────────────────────────
    def account_balance(self) -> float:
        resp = self._call({"balance": 1, "account": "current"})
        return float(resp.get("balance", {}).get("balance", 0))

    def account_info(self) -> dict:
        resp = self._call({"authorize": self.api_token})
        acct = resp.get("authorize", {})
        return {
            "balance":       float(acct.get("balance", 0)),
            "currency":      acct.get("currency", ""),
            "email":         acct.get("email", ""),
            "login_id":      acct.get("loginid", ""),
            "account_type":  self.account_type,
            "is_virtual":    acct.get("is_virtual", 0) == 1,
        }

    def get_accounts(self) -> List[dict]:
        """List all accounts (demo + real) linked to this token."""
        resp = self._call({"authorize": self.api_token})
        accounts = resp.get("authorize", {}).get("account_list", [])
        return [
            {
                "login_id":     a.get("loginid"),
                "account_type": "demo" if a.get("is_virtual") else "live",
                "currency":     a.get("currency"),
                "balance":      a.get("balance", 0),
            }
            for a in accounts
        ]

    def switch_account(self, login_id: str) -> bool:
        """Switch to a different account (demo ↔ live)."""
        resp = self._call({"switch_account": 1, "loginid": login_id})
        if "error" in resp:
            logger.error("Account switch failed: %s", resp["error"]["message"])
            return False
        logger.info("Switched to account: %s", login_id)
        return True

    # ── Market data ───────────────────────────────────────────────────────
    def get_price(self, symbol: str) -> Optional[Dict]:
        """Get latest tick for a symbol."""
        deriv_sym = DERIV_SYMBOLS.get(symbol, symbol)
        resp = self._call({"ticks": deriv_sym, "subscribe": 0})
        if "error" in resp:
            logger.debug("get_price error for %s: %s", symbol, resp["error"].get("message"))
            return None
        tick = resp.get("tick", {})
        bid  = float(tick.get("bid", 0))
        ask  = float(tick.get("ask", 0))
        if bid == 0 and ask == 0:
            # Some symbols only have quote
            quote = float(tick.get("quote", 0))
            bid = ask = quote
        return {
            "bid":    bid,
            "ask":    ask,
            "mid":    (bid + ask) / 2 if bid and ask else float(tick.get("quote", 0)),
            "spread": ask - bid,
            "time":   datetime.fromtimestamp(tick.get("epoch", time.time())),
            "symbol": symbol,
        }

    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: str,
        count:     int = 500,
        start:     Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle history from Deriv.
        Works for forex, metals, crypto AND synthetic indices.
        """
        deriv_sym   = DERIV_SYMBOLS.get(symbol, symbol)
        granularity = GRANULARITY.get(timeframe, 3600)

        if start is None:
            start = datetime.utcnow() - timedelta(seconds=granularity * count)

        start_epoch = int(start.timestamp())
        end_epoch   = int(datetime.utcnow().timestamp())

        all_candles = []
        chunk_size  = min(count, 5000)   # Deriv max per request

        request = {
            "ticks_history": deriv_sym,
            "adjust_start_time": 1,
            "count":        chunk_size,
            "end":          "latest",
            "start":        start_epoch,
            "style":        "candles",
            "granularity":  granularity,
        }

        resp = self._call(request, timeout=30)
        if "error" in resp:
            msg = resp["error"].get("message", "unknown")
            logger.error("get_ohlcv failed for %s: %s", symbol, msg)
            return pd.DataFrame()

        candles = resp.get("candles", [])
        if not candles:
            logger.warning("No candles returned for %s %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df = df.rename(columns={"open":"open","high":"high","low":"low","close":"close"})
        df["volume"] = 1.0   # Deriv doesn't provide tick volume — use 1 as placeholder
        df = df.set_index("datetime")[["open","high","low","close","volume"]]
        df = df.astype(float)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        logger.debug("Fetched %d candles for %s %s", len(df), symbol, timeframe)
        return df

    def get_symbols(self) -> List[dict]:
        """Get all available trading symbols."""
        resp = self._call({"active_symbols": "brief", "product_type": "basic"})
        symbols = resp.get("active_symbols", [])
        return [
            {
                "symbol":      s.get("symbol"),
                "display":     s.get("display_name"),
                "market":      s.get("market"),
                "submarket":   s.get("submarket"),
                "pip":         s.get("pip", 0.0001),
                "is_open":     s.get("exchange_is_open", False),
            }
            for s in symbols
        ]

    # ── Trade execution ───────────────────────────────────────────────────
    def get_contract_proposal(
        self,
        symbol:    str,
        direction: int,      # 2=BUY (CALL), 0=SELL (PUT)
        amount:    float,    # stake in account currency
        duration:  int = 5,  # contract duration
        duration_unit: str = "m",   # m=minutes, h=hours, d=days
        basis:     str = "stake",
    ) -> Optional[dict]:
        """
        Get price proposal for a Deriv contract.
        Deriv uses binary options / multipliers / CFD model.
        """
        deriv_sym    = DERIV_SYMBOLS.get(symbol, symbol)
        contract_type = "CALL" if direction == 2 else "PUT"

        resp = self._call({
            "proposal": 1,
            "amount":   amount,
            "basis":    basis,
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol":   deriv_sym,
        })
        if "error" in resp:
            logger.error("Proposal error: %s", resp["error"]["message"])
            return None
        return resp.get("proposal")

    def buy_contract(
        self,
        symbol:    str,
        direction: int,      # 2=BUY, 0=SELL
        amount:    float,    # stake amount
        duration:  int = 60,       # minutes
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        multiplier: Optional[float] = None,   # for CFD multiplier contracts
    ) -> Optional[dict]:
        """
        Execute a trade on Deriv.
        Returns contract details or None on failure.

        Two modes:
          Standard: binary options (duration-based)
          Multiplier: CFD-style with SL/TP (recommended for forex trading)
        """
        deriv_sym = DERIV_SYMBOLS.get(symbol, symbol)

        # ── Multiplier contract (CFD-style, recommended for forex) ─────────
        if multiplier is not None:
            limit_order = {}
            if stop_loss:
                limit_order["stop_loss"] = {"order_amount": stop_loss, "order_type": "stop_loss"}
            if take_profit:
                limit_order["take_profit"] = {"order_amount": take_profit, "order_type": "take_profit"}

            resp = self._call({
                "proposal": 1,
                "amount":   amount,
                "basis":    "stake",
                "contract_type": "MULTUP" if direction == 2 else "MULTDOWN",
                "currency": "USD",
                "symbol":   deriv_sym,
                "multiplier": multiplier,
                **({"limit_order": limit_order} if limit_order else {}),
            })
            if "error" in resp or "proposal" not in resp:
                logger.error("Multiplier proposal failed: %s",
                             resp.get("error", {}).get("message", "unknown"))
                return None

            buy_resp = self._call({
                "buy":   resp["proposal"]["id"],
                "price": resp["proposal"]["ask_price"],
            })

        # ── Standard binary / duration-based contract ──────────────────────
        else:
            contract_type = "CALL" if direction == 2 else "PUT"
            proposal_resp = self._call({
                "proposal":  1,
                "amount":    amount,
                "basis":     "stake",
                "contract_type": contract_type,
                "currency":  "USD",
                "duration":  duration,
                "duration_unit": "m",
                "symbol":    deriv_sym,
            })
            if "error" in proposal_resp:
                logger.error("Proposal error: %s",
                             proposal_resp["error"]["message"])
                return None
            proposal = proposal_resp.get("proposal", {})
            buy_resp  = self._call({
                "buy":   proposal["id"],
                "price": proposal["ask_price"],
            })

        if "error" in buy_resp:
            logger.error("Buy failed: %s", buy_resp["error"]["message"])
            return None

        contract = buy_resp.get("buy", {})
        logger.info(
            "✅ CONTRACT BOUGHT | %s %s | amount=%.2f | contract_id=%s | price=%.5f",
            "BUY" if direction==2 else "SELL", symbol,
            amount, contract.get("contract_id"), float(contract.get("buy_price", 0))
        )
        return contract

    def sell_contract(self, contract_id: int, price: float = 0) -> Optional[dict]:
        """Close/sell an open contract early."""
        resp = self._call({"sell": contract_id, "price": price})
        if "error" in resp:
            logger.error("Sell failed: %s", resp["error"]["message"])
            return None
        result = resp.get("sell", {})
        logger.info(
            "Contract sold: id=%d | sold_for=%.2f | profit=%.2f",
            contract_id,
            float(result.get("sold_for", 0)),
            float(result.get("profit", 0))
        )
        return result

    # ── For-profit table & open positions ─────────────────────────────────
    def get_open_contracts(self) -> List[dict]:
        resp = self._call({"portfolio": 1})
        contracts = resp.get("portfolio", {}).get("contracts", [])
        return [
            {
                "id":           c.get("contract_id"),
                "symbol":       DERIV_TO_SYMBOL.get(c.get("underlying"), c.get("underlying")),
                "direction":    "BUY" if c.get("contract_type") in ("CALL","MULTUP") else "SELL",
                "buy_price":    float(c.get("buy_price", 0)),
                "current_value": float(c.get("bid_price", 0)),
                "pnl":          float(c.get("bid_price", 0)) - float(c.get("buy_price", 0)),
                "expiry":       c.get("expiry_time"),
            }
            for c in contracts
        ]

    def get_trade_history(self, count: int = 50) -> List[dict]:
        resp = self._call({
            "profit_table": 1,
            "description":  1,
            "limit":        count,
            "sort":         "DESC",
        })
        trades = resp.get("profit_table", {}).get("transactions", [])
        return [
            {
                "id":        t.get("contract_id"),
                "symbol":    DERIV_TO_SYMBOL.get(t.get("underlying"), t.get("underlying")),
                "buy_price": float(t.get("buy_price", 0)),
                "sell_price": float(t.get("sell_price", 0)),
                "pnl":       float(t.get("sell_price", 0)) - float(t.get("buy_price", 0)),
                "buy_time":  datetime.fromtimestamp(t.get("purchase_time", 0)),
                "sell_time": datetime.fromtimestamp(t.get("sell_time", 0)) if t.get("sell_time") else None,
            }
            for t in trades
        ]

    # ── Compatibility shims (AutoTrader interface) ─────────────────────────
    def connect_broker(self) -> bool:
        return self.connect()

    def place_market_order(
        self,
        symbol: str,
        direction: int,
        units: float,
        sl: float = 0,
        tp: float = 0,
        comment: str = "AI bot",
    ) -> Optional[str]:
        """AutoTrader-compatible interface — maps to buy_contract."""
        contract = self.buy_contract(
            symbol      = symbol,
            direction   = direction,
            amount      = units,      # units = stake amount for Deriv
            multiplier  = 100,        # CFD multiplier (adjust per symbol)
            stop_loss   = sl if sl else None,
            take_profit = tp if tp else None,
        )
        if contract:
            return str(contract.get("contract_id"))
        return None

    def close_trade(self, trade_id) -> Optional[float]:
        result = self.sell_contract(int(trade_id))
        if result:
            return float(result.get("profit", 0))
        return None

    def get_open_trades(self) -> List[dict]:
        """AutoTrader-compatible open positions."""
        contracts = self.get_open_contracts()
        return [
            {
                "id":           str(c["id"]),
                "symbol":       c["symbol"],
                "direction":    c["direction"],
                "units":        c["buy_price"],
                "lot_size":     c["buy_price"],
                "entry_price":  c["buy_price"],
                "sl":           0,
                "tp":           0,
                "unrealized_pnl": c["pnl"],
                "open_time":    datetime.utcnow().isoformat(),
            }
            for c in contracts
        ]

    def modify_trade_sl(self, trade_id, new_sl: float) -> bool:
        logger.debug("Deriv: trailing SL update not supported for open contracts")
        return False

    @staticmethod
    def lot_to_units(symbol: str, lot_size: float) -> float:
        """For Deriv, lot_size = stake amount in account currency."""
        return max(1.0, lot_size * 100)   # map 0.01 lot → $1 stake minimum

    # ── WebSocket I/O ─────────────────────────────────────────────────────
    def _call(self, payload: dict, timeout: int = None) -> dict:
        """Send a request and wait for the matching response."""
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() first")
        self._req_id += 1
        payload["req_id"] = self._req_id
        to = timeout or self._timeout

        for attempt in range(3):
            try:
                self._ws.send(json.dumps(payload))
                deadline = time.time() + to
                while time.time() < deadline:
                    raw  = self._ws.recv()
                    resp = json.loads(raw)
                    if resp.get("req_id") == self._req_id:
                        return resp
                # Timeout
                logger.warning("Timeout waiting for req_id=%d (attempt %d)",
                                self._req_id, attempt+1)
            except websocket.WebSocketConnectionClosedException:
                logger.warning("WS closed — reconnecting (attempt %d)", attempt+1)
                if not self.reconnect():
                    break
                payload["req_id"] = self._req_id   # keep same req_id
            except Exception as exc:
                logger.warning("WS call error (attempt %d): %s", attempt+1, exc)
                if not self.reconnect():
                    break
                time.sleep(1)

        return {"error": {"message": f"Request failed after 3 attempts"}}


# ── Real-time tick streamer (background thread) ───────────────────────────

class DerivTickStreamer:
    """
    Streams live ticks for multiple symbols in a background thread.
    Access latest tick via: streamer.get_latest(symbol)
    """

    def __init__(self, api_token: str, symbols: List[str]):
        self.api_token = api_token
        self.symbols   = symbols
        self._latest:  Dict[str, dict] = {}
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Tick streamer started for %s", self.symbols)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_latest(self, symbol: str) -> Optional[dict]:
        return self._latest.get(symbol)

    def on_tick(self, callback: Callable) -> None:
        """Register a callback: callback(symbol, tick_dict)"""
        self._callbacks.append(callback)

    def _run(self) -> None:
        import websocket as ws

        def on_message(wsapp, message):
            data = json.loads(message)
            if data.get("msg_type") == "tick":
                tick = data["tick"]
                sym  = DERIV_TO_SYMBOL.get(tick["symbol"], tick["symbol"])
                entry = {
                    "symbol": sym,
                    "bid":    tick.get("bid", tick.get("quote", 0)),
                    "ask":    tick.get("ask", tick.get("quote", 0)),
                    "mid":    tick.get("quote", 0),
                    "time":   datetime.fromtimestamp(tick.get("epoch", time.time())),
                }
                self._latest[sym] = entry
                for cb in self._callbacks:
                    try: cb(sym, entry)
                    except Exception as exc: logger.warning("Tick callback error: %s", exc)

        def on_open(wsapp):
            # Authorize first
            wsapp.send(json.dumps({"authorize": self.api_token, "req_id": 0}))
            # Subscribe to ticks for each symbol
            for i, sym in enumerate(self.symbols, start=1):
                deriv_sym = DERIV_SYMBOLS.get(sym, sym)
                wsapp.send(json.dumps({
                    "ticks":     deriv_sym,
                    "subscribe": 1,
                    "req_id":    i,
                }))

        def on_error(wsapp, error):
            logger.warning("Tick stream error: %s", error)

        def on_close(wsapp, close_status_code, close_msg):
            logger.warning("Tick stream closed (code=%s)", close_status_code)

        # Iterative reconnect loop — avoids unbounded recursion on disconnect.
        while self._running:
            wsapp = ws.WebSocketApp(
                DERIV_WS_URL,
                on_message = on_message,
                on_open    = on_open,
                on_error   = on_error,
                on_close   = on_close,
            )
            wsapp.run_forever(ping_interval=30, ping_timeout=10)
            if self._running:
                logger.warning("Tick stream disconnected — reconnecting in 5s")
                time.sleep(5)


# ── Convenience factory ───────────────────────────────────────────────────

def create_deriv_client(mode: str = "demo") -> "DerivClient":
    """
    Create a DerivClient from environment variables.
    mode = "demo" → uses DERIV_API_TOKEN_DEMO
    mode = "live" → uses DERIV_API_TOKEN_LIVE
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    if mode == "demo":
        token = os.getenv("DERIV_API_TOKEN_DEMO", "")
    else:
        token = os.getenv("DERIV_API_TOKEN_LIVE", "")

    if not token:
        raise ValueError(
            f"DERIV_API_TOKEN_{mode.upper()} not set in .env\n"
            "Get your token at: app.deriv.com/account/api-token"
        )
    return DerivClient(token, account_type=mode)
