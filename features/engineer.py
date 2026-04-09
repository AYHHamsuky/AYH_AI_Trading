"""
Feature engineering — Technical indicators + lag features + labels
Returns a ready-to-train DataFrame.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from config import FEATURE_CONFIG as FC

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Computes all technical indicators and generates supervised-learning labels.
    Call build(df) on raw OHLCV data.
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or FC

    # ── Main entry point ──────────────────────────────────────────────────
    def build(
        self,
        df: pd.DataFrame,
        add_labels: bool = True,
        add_smc: bool = True,
    ) -> pd.DataFrame:
        """Return feature DataFrame (and optionally SMC + classification labels)."""
        out = df.copy()
        out = self._trend_indicators(out)
        out = self._momentum_indicators(out)
        out = self._volatility_indicators(out)
        out = self._volume_indicators(out)
        out = self._lag_features(out)
        out = self._return_features(out)
        out = self._time_features(out)
        if add_smc:
            out = self._smc_features(out)   # ← SMC signals as XGBoost features
        if add_labels:
            out = self._make_labels(out)
        out.dropna(inplace=True)
        logger.debug("Features built: %d rows × %d cols", len(out), len(out.columns))
        return out

    def feature_names(self, df: pd.DataFrame) -> list:
        built = self.build(df.copy(), add_labels=False)
        return [c for c in built.columns if c not in ("open","high","low","close","volume")]

    # ── Trend ─────────────────────────────────────────────────────────────
    def _trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # EMAs
        for p in self.cfg["ema_periods"]:
            df[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

        # EMA slopes
        df["ema_slope_9"]  = df["ema_9"].diff(3)  / df["ema_9"].shift(3)
        df["ema_slope_21"] = df["ema_21"].diff(3) / df["ema_21"].shift(3)

        # EMA cross signals
        df["ema9_21_cross"]  = np.sign(df["ema_9"]  - df["ema_21"])
        df["ema21_50_cross"] = np.sign(df["ema_21"] - df["ema_50"])

        # MACD
        ema_fast   = close.ewm(span=self.cfg["macd_fast"],   adjust=False).mean()
        ema_slow   = close.ewm(span=self.cfg["macd_slow"],   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.cfg["macd_signal"], adjust=False).mean()
        df["macd"]         = macd_line
        df["macd_signal"]  = signal_line
        df["macd_hist"]    = macd_line - signal_line
        df["macd_cross"]   = np.sign(macd_line - signal_line)

        # ADX (Average Directional Index)
        df["adx"] = self._adx(high, low, close, period=14)

        # Ichimoku components
        df["ichimoku_conversion"] = (high.rolling(9).max()  + low.rolling(9).min())  / 2
        df["ichimoku_base"]       = (high.rolling(26).max() + low.rolling(26).min()) / 2
        df["ichimoku_diff"]       = df["ichimoku_conversion"] - df["ichimoku_base"]

        return df

    # ── Momentum ──────────────────────────────────────────────────────────
    def _momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        p     = self.cfg["rsi_period"]

        # RSI
        df["rsi"] = self._rsi(close, p)
        df["rsi_overbought"]  = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"]    = (df["rsi"] < 30).astype(int)

        # Stochastic
        df["stoch_k"], df["stoch_d"] = self._stochastic(high, low, close)
        df["stoch_cross"] = np.sign(df["stoch_k"] - df["stoch_d"])

        # CCI
        tp = (high + low + close) / 3
        df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        # Williams %R
        highest_high = high.rolling(14).max()
        lowest_low   = low.rolling(14).min()
        df["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)

        # Rate of Change
        df["roc_5"]  = close.pct_change(5)  * 100
        df["roc_10"] = close.pct_change(10) * 100

        # Momentum
        df["momentum_10"] = close - close.shift(10)

        return df

    # ── Volatility ────────────────────────────────────────────────────────
    def _volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ATR
        df["atr"] = self._atr(high, low, close, self.cfg["atr_period"])
        df["atr_pct"] = df["atr"] / close * 100

        # Bollinger Bands
        ma   = close.rolling(self.cfg["bb_period"]).mean()
        std  = close.rolling(self.cfg["bb_period"]).std()
        df["bb_upper"]  = ma + self.cfg["bb_std"] * std
        df["bb_lower"]  = ma - self.cfg["bb_std"] * std
        df["bb_middle"] = ma
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / ma
        df["bb_pct"]    = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.1)).astype(int)

        # Historical volatility (20-bar)
        df["hv_20"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

        # Keltner Channels
        ema20 = close.ewm(span=20, adjust=False).mean()
        df["kc_upper"] = ema20 + 1.5 * df["atr"]
        df["kc_lower"] = ema20 - 1.5 * df["atr"]
        df["kc_pct"]   = (close - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"] + 1e-9)

        return df

    # ── Volume ────────────────────────────────────────────────────────────
    def _volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns or df["volume"].sum() == 0:
            return df
        vol   = df["volume"]
        close = df["close"]

        df["volume_sma10"] = vol.rolling(10).mean()
        df["volume_sma20"] = vol.rolling(20).mean()
        df["volume_ratio"] = vol / (df["volume_sma20"] + 1e-9)

        # OBV
        direction = np.sign(close.diff())
        df["obv"]  = (vol * direction).cumsum()

        # VWAP (rolling 20)
        tp = (df["high"] + df["low"] + close) / 3
        df["vwap"]     = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
        df["vwap_dist"] = (close - df["vwap"]) / df["vwap"]

        return df

    # ── Lag & return features ─────────────────────────────────────────────
    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["close", "rsi", "macd", "atr_pct"]:
            if col not in df.columns:
                continue
            for lag in self.cfg["lag_periods"]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df

    def _return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        for p in self.cfg["return_periods"]:
            df[f"ret_{p}"]     = close.pct_change(p)
            df[f"log_ret_{p}"] = np.log(close / close.shift(p))
        return df

    # ── Time / session features ───────────────────────────────────────────
    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = pd.to_datetime(df.index)
        df["hour"]        = idx.hour
        df["dow"]         = idx.dayofweek          # 0=Mon, 4=Fri
        df["is_london"]   = ((idx.hour >= 7)  & (idx.hour < 16)).astype(int)
        df["is_newyork"]  = ((idx.hour >= 12) & (idx.hour < 21)).astype(int)
        df["is_session_overlap"] = (df["is_london"] & df["is_newyork"]).astype(int)
        return df

    # ── Label generation ──────────────────────────────────────────────────
    def _make_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        horizon = self.cfg["target_horizon"]
        pct     = self.cfg["target_pct"]
        future_ret = df["close"].pct_change(horizon).shift(-horizon)
        df["label"] = 1  # HOLD
        df.loc[future_ret >  pct, "label"] = 2   # BUY
        df.loc[future_ret < -pct, "label"] = 0   # SELL
        df["future_return"] = future_ret
        return df

    # ── Technical helper functions ────────────────────────────────────────
    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        up   = high.diff()
        down = -low.diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        tr = FeatureEngineer._atr(high, low, close, period)
        plus_di  = pd.Series(plus_dm,  index=close.index).ewm(com=period-1, adjust=False).mean() / (tr + 1e-9) * 100
        minus_di = pd.Series(minus_dm, index=close.index).ewm(com=period-1, adjust=False).mean() / (tr + 1e-9) * 100
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
        return dx.ewm(com=period-1, adjust=False).mean()

    @staticmethod
    def _stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low   = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
        d = k.rolling(d_period).mean()
        return k, d

    # ── SMC feature injection ─────────────────────────────────────────────
    def _smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run SMC analysis on a rolling window and encode results as
        numeric features for XGBoost. Each row gets its own snapshot
        so there is NO lookahead — only past data is used per bar.
        """
        try:
            from features.smc import SMCEngine
        except ImportError:
            logger.warning("SMC engine not found — skipping SMC features")
            return df

        engine  = SMCEngine(swing_lookback=8)
        n       = len(df)
        min_bars = 50   # need at least this many bars for SMC

        # Pre-allocate feature arrays
        smc_bias_bull         = np.zeros(n)
        smc_bias_bear         = np.zeros(n)
        smc_bias_conf         = np.full(n, 0.5)
        smc_ob_bull_near      = np.zeros(n)
        smc_ob_bear_near      = np.zeros(n)
        smc_ob_bull_count     = np.zeros(n)
        smc_ob_bear_count     = np.zeros(n)
        smc_sweep_bull_recent = np.zeros(n)
        smc_sweep_bear_recent = np.zeros(n)
        smc_sweep_count       = np.zeros(n)
        smc_inducement_bull   = np.zeros(n)
        smc_inducement_bear   = np.zeros(n)
        smc_fvg_bull_near     = np.zeros(n)
        smc_fvg_bear_near     = np.zeros(n)
        smc_signal_dir        = np.ones(n)    # default HOLD=1
        smc_signal_conf       = np.full(n, 0.5)

        # Use raw OHLCV columns only for SMC (avoid feature leakage)
        ohlcv_cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]

        # Rolling window — step every 5 bars for speed, interpolate between
        step = 5
        last_result = None
        for i in range(min_bars, n, step):
            window = df[ohlcv_cols].iloc[:i].copy()
            try:
                result = engine.analyze(window)
                signal = engine.get_smc_signal(window)
                last_result = (result, signal)
            except Exception:
                if last_result is None:
                    continue
                result, signal = last_result

            # Fill from i to i+step
            end = min(i + step, n)
            close_now = df["close"].iloc[i - 1]
            atr_now   = df["atr"].iloc[i - 1] if "atr" in df.columns else close_now * 0.001

            # Bias
            is_bull = result.bias == "bullish"
            is_bear = result.bias == "bearish"
            smc_bias_bull[i:end] = float(is_bull)
            smc_bias_bear[i:end] = float(is_bear)
            smc_bias_conf[i:end] = result.bias_confidence

            # Active order blocks
            active_bull_obs = [ob for ob in result.order_blocks if ob.ob_type=="bullish" and ob.active]
            active_bear_obs = [ob for ob in result.order_blocks if ob.ob_type=="bearish" and ob.active]
            smc_ob_bull_count[i:end] = len(active_bull_obs)
            smc_ob_bear_count[i:end] = len(active_bear_obs)

            # Price near OB (within 2×ATR)
            near_bull = any(abs(close_now - ob.midpoint) < atr_now * 2 for ob in active_bull_obs)
            near_bear = any(abs(close_now - ob.midpoint) < atr_now * 2 for ob in active_bear_obs)
            smc_ob_bull_near[i:end] = float(near_bull)
            smc_ob_bear_near[i:end] = float(near_bear)

            # Recent sweeps (last 5 bars)
            recent_sweeps = [s for s in result.liquidity_sweeps
                             if s.confirmed and (i - s.index) <= 5]
            smc_sweep_count[i:end] = len(recent_sweeps)
            smc_sweep_bull_recent[i:end] = float(any(s.sweep_type=="sellside" for s in recent_sweeps))
            smc_sweep_bear_recent[i:end] = float(any(s.sweep_type=="buyside"  for s in recent_sweeps))

            # Recent swept inducements
            recent_inds = [ind for ind in result.inducements
                           if ind.swept and ind.swept_index is not None and (i - ind.swept_index) <= 10]
            smc_inducement_bull[i:end] = float(any(ind.ind_type=="bullish" for ind in recent_inds))
            smc_inducement_bear[i:end] = float(any(ind.ind_type=="bearish" for ind in recent_inds))

            # Nearby FVGs (unfilled, within 3×ATR)
            smc_fvg_bull_near[i:end] = float(any(
                fvg.fvg_type=="bullish" and abs(close_now - fvg.midpoint) < atr_now * 3
                for fvg in result.fair_value_gaps
            ))
            smc_fvg_bear_near[i:end] = float(any(
                fvg.fvg_type=="bearish" and abs(close_now - fvg.midpoint) < atr_now * 3
                for fvg in result.fair_value_gaps
            ))

            # SMC signal
            smc_signal_dir[i:end]  = signal["signal"]
            smc_signal_conf[i:end] = signal["confidence"]

        # Assign to DataFrame
        df["smc_bias_bull"]         = smc_bias_bull
        df["smc_bias_bear"]         = smc_bias_bear
        df["smc_bias_conf"]         = smc_bias_conf
        df["smc_ob_bull_near"]      = smc_ob_bull_near
        df["smc_ob_bear_near"]      = smc_ob_bear_near
        df["smc_ob_bull_count"]     = smc_ob_bull_count
        df["smc_ob_bear_count"]     = smc_ob_bear_count
        df["smc_sweep_bull_recent"] = smc_sweep_bull_recent
        df["smc_sweep_bear_recent"] = smc_sweep_bear_recent
        df["smc_sweep_count"]       = smc_sweep_count
        df["smc_inducement_bull"]   = smc_inducement_bull
        df["smc_inducement_bear"]   = smc_inducement_bear
        df["smc_fvg_bull_near"]     = smc_fvg_bull_near
        df["smc_fvg_bear_near"]     = smc_fvg_bear_near
        df["smc_signal_dir"]        = smc_signal_dir
        df["smc_signal_conf"]       = smc_signal_conf

        logger.info("SMC features added: 16 columns injected into training frame")
        return df
