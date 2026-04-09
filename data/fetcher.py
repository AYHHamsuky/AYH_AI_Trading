"""
Unified Data Fetcher
────────────────────
Priority:
  1. Deriv API  — primary (real-time + history for all symbols including synthetics)
  2. yfinance   — fallback for forex/metals/crypto when Deriv token not set
  3. CCXT       — fallback for crypto

Usage:
  fetcher = DataFetcher()
  df = fetcher.fetch("EURUSD", "H1", bars=500)
  df = fetcher.fetch("V75",    "M5", bars=500)   # Volatility 75 (Deriv synthetic)
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# yfinance symbol mapping
YF_SYMBOL_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "XAUUSD": "GC=F",     "XAGUSD": "SI=F",
    "BTCUSD": "BTC-USD",  "ETHUSD": "ETH-USD",
}

# yfinance interval mapping
YF_TF_MAP = {
    "M1":"1m","M5":"5m","M15":"15m","M30":"30m",
    "H1":"1h","H4":"4h","D1":"1d",
}

# Synthetic indices only available on Deriv
DERIV_ONLY = {"V10","V25","V50","V75","V100","BOOM1000","CRASH1000","JUMP10","JUMP25","JUMP50"}


class DataFetcher:
    """
    Fetches OHLCV data from Deriv API (primary) or yfinance (fallback).
    Auto-detects which source to use per symbol.
    """

    def __init__(
        self,
        deriv_token: Optional[str] = None,
        deriv_mode:  str = "demo",     # "demo" | "live"
        fallback:    str = "yfinance", # "yfinance" | "ccxt"
    ):
        # Try to load Deriv token from env if not provided
        if deriv_token is None:
            from dotenv import load_dotenv; load_dotenv()
            key = f"DERIV_API_TOKEN_{deriv_mode.upper()}"
            deriv_token = os.getenv(key, "")

        self.deriv_token = deriv_token
        self.deriv_mode  = deriv_mode
        self.fallback    = fallback
        self._deriv:     Optional["DerivClient"] = None
        self._deriv_ok   = False

        if self.deriv_token:
            self._init_deriv()

    def _init_deriv(self) -> None:
        try:
            from execution.deriv_broker import DerivClient
            self._deriv = DerivClient(self.deriv_token, self.deriv_mode)
            self._deriv_ok = self._deriv.connect()
            if self._deriv_ok:
                logger.info("Deriv data source connected [%s]", self.deriv_mode.upper())
            else:
                logger.warning("Deriv connection failed (timeout/network issue) — using yfinance fallback")
        except ImportError:
            logger.warning("deriv_broker not available — using yfinance")
        except Exception as exc:
            logger.warning("Deriv initialization error: %s — using yfinance fallback", exc)

    # ── Public API ────────────────────────────────────────────────────────
    def fetch(
        self,
        symbol:    str,
        timeframe: str,
        bars:      int = 500,
        start:     Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. Returns clean DataFrame with DatetimeIndex.
        Automatically chooses Deriv or yfinance based on symbol and availability.
        """
        # Synthetic indices MUST use Deriv
        if symbol in DERIV_ONLY:
            if not self._deriv_ok:
                raise RuntimeError(
                    f"{symbol} requires Deriv API connection.\n"
                    f"• Token in .env: {bool(self.deriv_token)}\n"
                    f"• Connection status: {'✓ OK' if self._deriv_ok else '✗ FAILED'}\n"
                    f"• Check: Network connectivity, API token scopes (Trade + Read + Admin), rate limits"
                )
            return self._fetch_deriv(symbol, timeframe, bars, start)

        # For standard instruments: try Deriv first, fallback to yfinance
        if self._deriv_ok:
            try:
                df = self._fetch_deriv(symbol, timeframe, bars, start)
                if len(df) >= 10:
                    return df
                logger.debug("Deriv returned too few bars for %s — trying fallback", symbol)
            except Exception as exc:
                logger.debug("Deriv fetch failed for %s: %s — falling back", symbol, exc)

        return self._fetch_fallback(symbol, timeframe, bars, start)

    def fetch_realtime_price(self, symbol: str) -> Optional[dict]:
        """Get latest bid/ask/mid for a symbol (Deriv only)."""
        if not self._deriv_ok:
            return None
        return self._deriv.get_price(symbol)

    def available_symbols(self) -> list:
        """Return list of available symbols."""
        if self._deriv_ok:
            try:
                syms = self._deriv.get_symbols()
                return [s["symbol"] for s in syms if s.get("is_open")]
            except Exception:
                pass
        return list(YF_SYMBOL_MAP.keys())

    def source(self) -> str:
        return "deriv" if self._deriv_ok else self.fallback

    # ── Deriv ─────────────────────────────────────────────────────────────
    def _fetch_deriv(
        self, symbol: str, timeframe: str,
        bars: int, start: Optional[datetime]
    ) -> pd.DataFrame:
        if start is None:
            from execution.deriv_broker import GRANULARITY
            gran  = GRANULARITY.get(timeframe, 3600)
            start = datetime.utcnow() - timedelta(seconds=gran * (bars + 10))
        df = self._deriv.get_ohlcv(symbol, timeframe, count=bars, start=start)
        return self._clean(df)

    # ── yfinance fallback ─────────────────────────────────────────────────
    def _fetch_fallback(
        self, symbol: str, timeframe: str,
        bars: int, start: Optional[datetime]
    ) -> pd.DataFrame:
        if self.fallback == "ccxt" and symbol in {"BTCUSD","ETHUSD","LTCUSD"}:
            return self._fetch_ccxt(symbol, timeframe, bars, start)
        return self._fetch_yfinance(symbol, timeframe, bars, start)

    def _fetch_yfinance(
        self, symbol: str, timeframe: str,
        bars: int, start: Optional[datetime]
    ) -> pd.DataFrame:
        import yfinance as yf
        yf_sym   = YF_SYMBOL_MAP.get(symbol, symbol)
        interval = YF_TF_MAP.get(timeframe, "1h")
        if start is None:
            tf_mins = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440}
            mins    = tf_mins.get(timeframe, 60) * bars
            start   = datetime.utcnow() - timedelta(minutes=mins)
        df = yf.download(yf_sym, start=start, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"yfinance returned no data for {yf_sym}")
        df.columns = [c.lower() for c in df.columns]
        df["volume"] = df.get("volume", pd.Series(1.0, index=df.index))
        return self._clean(df[["open","high","low","close","volume"]])

    def _fetch_ccxt(
        self, symbol: str, timeframe: str,
        bars: int, start: Optional[datetime]
    ) -> pd.DataFrame:
        try:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True})
            ccxt_sym = symbol.replace("USD", "/USDT")
            tf_map   = {"M1":"1m","M5":"5m","M15":"15m","H1":"1h","H4":"4h","D1":"1d"}
            tf_str   = tf_map.get(timeframe, "1h")
            since    = int(start.timestamp() * 1000) if start else None
            ohlcv    = exchange.fetch_ohlcv(ccxt_sym, tf_str, since=since, limit=bars)
            df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
            return self._clean(df.set_index("datetime")[["open","high","low","close","volume"]])
        except Exception as exc:
            logger.warning("CCXT failed for %s: %s — trying yfinance", symbol, exc)
            return self._fetch_yfinance(symbol, timeframe, bars, start)

    # ── Cleaning ──────────────────────────────────────────────────────────
    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        else:
            df.index = df.index.tz_localize("UTC").tz_convert(None)
        df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns:
                df = df[df[col] > 0]
        df = df[df["high"] >= df["low"]]
        df = df[~df.index.duplicated(keep="last")]
        df.dropna(subset=["open","high","low","close"], inplace=True)
        if "volume" in df.columns:
            df["volume"] = df["volume"].replace(0, 1).fillna(1)
        return df
