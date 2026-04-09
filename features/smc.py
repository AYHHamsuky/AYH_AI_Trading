"""
Smart Money Concepts (SMC) Engine
─────────────────────────────────
Detects:
  1. Order Blocks (OB)       — Last opposing candle before impulse move
  2. Sweep of Liquidity      — Price takes out swing high/low then reverses
  3. Inducement              — Minor liquidity pool swept before main move
  4. Break of Structure (BOS)— Confirms trend direction post-sweep
  5. Fair Value Gaps (FVG)   — Imbalance zones price tends to fill
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    index:      int
    timestamp:  pd.Timestamp
    ob_type:    str          # "bullish" | "bearish"
    high:       float
    low:        float
    open:       float
    close:      float
    impulse_size: float      # strength of the following impulse
    mitigated:  bool = False # True once price returns into OB
    active:     bool = True

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2

    @property
    def size(self) -> float:
        return self.high - self.low


@dataclass
class LiquiditySweep:
    index:        int
    timestamp:    pd.Timestamp
    sweep_type:   str        # "buyside" | "sellside"
    level:        float      # swept price level
    sweep_high:   float
    sweep_low:    float
    reversal_size: float     # how far price reversed after the sweep
    confirmed:    bool = False
    inducement:   bool = False  # True if this was an inducement sweep


@dataclass
class Inducement:
    index:        int
    timestamp:    pd.Timestamp
    ind_type:     str        # "bullish" | "bearish"
    level:        float
    swept:        bool = False
    swept_index:  Optional[int] = None


@dataclass
class FairValueGap:
    index:        int
    timestamp:    pd.Timestamp
    fvg_type:     str        # "bullish" | "bearish"
    high:         float
    low:          float
    filled:       bool = False

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class SMCResult:
    order_blocks:      List[OrderBlock]      = field(default_factory=list)
    liquidity_sweeps:  List[LiquiditySweep]  = field(default_factory=list)
    inducements:       List[Inducement]       = field(default_factory=list)
    fair_value_gaps:   List[FairValueGap]     = field(default_factory=list)
    swing_highs:       List[tuple]            = field(default_factory=list)  # (index, price)
    swing_lows:        List[tuple]            = field(default_factory=list)
    bias:              str                    = "neutral"   # "bullish" | "bearish" | "neutral"
    bias_confidence:   float                  = 0.5


# ── Main SMC Engine ────────────────────────────────────────────────────────

class SMCEngine:
    """
    Detects Smart Money Concept structures on any OHLCV DataFrame.

    Usage:
        engine = SMCEngine()
        result = engine.analyze(df)
    """

    def __init__(
        self,
        swing_lookback:    int   = 10,     # bars each side for swing detection
        impulse_threshold: float = 0.002,  # min move (0.2%) to qualify as impulse
        sweep_tolerance:   float = 0.0005, # how close price must get to level
        min_ob_size:       float = 0.0003, # minimum OB body size
        max_ob_age:        int   = 100,    # discard OBs older than N bars
    ):
        self.swing_lookback    = swing_lookback
        self.impulse_threshold = impulse_threshold
        self.sweep_tolerance   = sweep_tolerance
        self.min_ob_size       = min_ob_size
        self.max_ob_age        = max_ob_age

    def analyze(self, df: pd.DataFrame) -> SMCResult:
        """Run full SMC analysis on OHLCV DataFrame. Returns SMCResult."""
        if len(df) < self.swing_lookback * 3:
            return SMCResult()

        result = SMCResult()

        # Step 1: Identify swing highs / lows
        result.swing_highs, result.swing_lows = self._detect_swings(df)

        # Step 2: Detect order blocks
        result.order_blocks = self._detect_order_blocks(df)

        # Step 3: Detect liquidity sweeps
        result.liquidity_sweeps = self._detect_sweeps(df, result.swing_highs, result.swing_lows)

        # Step 4: Detect inducement (minor sweeps before major moves)
        result.inducements = self._detect_inducement(df, result.swing_highs, result.swing_lows)

        # Step 5: Fair Value Gaps
        result.fair_value_gaps = self._detect_fvg(df)

        # Step 6: Mark mitigated OBs
        self._check_ob_mitigation(df, result.order_blocks)

        # Step 7: Determine market bias
        result.bias, result.bias_confidence = self._determine_bias(
            df, result.order_blocks, result.liquidity_sweeps
        )

        logger.debug(
            "SMC: %d OBs | %d sweeps | %d inducements | %d FVGs | bias=%s",
            len(result.order_blocks), len(result.liquidity_sweeps),
            len(result.inducements), len(result.fair_value_gaps), result.bias
        )
        return result

    # ── Swing detection ───────────────────────────────────────────────────

    def _detect_swings(self, df: pd.DataFrame):
        lb = self.swing_lookback
        win = 2 * lb + 1

        high_series = df["high"]
        low_series  = df["low"]

        # Rolling max/min centred on each bar via a forward-shifted rolling window.
        # center=True is available in pandas ≥ 1.3 for min_periods; we compute it
        # manually to stay compatible: roll forward lb bars, then shift back lb.
        roll_max = high_series.rolling(win, min_periods=win).max().shift(-lb)
        roll_min = low_series.rolling(win, min_periods=win).min().shift(-lb)

        is_swing_high = (high_series == roll_max)
        is_swing_low  = (low_series  == roll_min)

        highs = [(i, high_series.iloc[i]) for i in is_swing_high[is_swing_high].index
                 if isinstance(i, int)]
        # iloc-based indexing for integer positions
        sh_idx = [df.index.get_loc(idx) for idx in is_swing_high[is_swing_high].index]
        sl_idx = [df.index.get_loc(idx) for idx in is_swing_low[is_swing_low].index]

        highs = [(i, high_series.iloc[i]) for i in sh_idx]
        lows  = [(i, low_series.iloc[i])  for i in sl_idx]
        return highs, lows

    # ── Order Blocks ──────────────────────────────────────────────────────

    def _detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        obs  = []
        n    = len(df)
        body = (df["close"] - df["open"]).abs()

        for i in range(1, n - 3):
            # Check for impulse move starting at i+1
            impulse_up   = self._measure_impulse(df, i + 1, direction=1)
            impulse_down = self._measure_impulse(df, i + 1, direction=-1)

            candle_open  = df["open"].iloc[i]
            candle_close = df["close"].iloc[i]
            candle_high  = df["high"].iloc[i]
            candle_low   = df["low"].iloc[i]
            candle_body  = abs(candle_close - candle_open) / (candle_close + 1e-9)

            if candle_body < self.min_ob_size:
                continue

            # Bullish OB: bearish candle before bullish impulse
            if (candle_close < candle_open                    # bearish candle
                    and impulse_up >= self.impulse_threshold):
                obs.append(OrderBlock(
                    index       = i,
                    timestamp   = pd.to_datetime(df.index[i]),
                    ob_type     = "bullish",
                    high        = candle_high,
                    low         = candle_low,
                    open        = candle_open,
                    close       = candle_close,
                    impulse_size= impulse_up,
                ))

            # Bearish OB: bullish candle before bearish impulse
            elif (candle_close > candle_open                  # bullish candle
                    and impulse_down >= self.impulse_threshold):
                obs.append(OrderBlock(
                    index       = i,
                    timestamp   = pd.to_datetime(df.index[i]),
                    ob_type     = "bearish",
                    high        = candle_high,
                    low         = candle_low,
                    open        = candle_open,
                    close       = candle_close,
                    impulse_size= impulse_down,
                ))

        # Keep only the most recent N per type, strongest first
        bull_obs = sorted([o for o in obs if o.ob_type == "bullish"],
                          key=lambda x: x.impulse_size, reverse=True)[:8]
        bear_obs = sorted([o for o in obs if o.ob_type == "bearish"],
                          key=lambda x: x.impulse_size, reverse=True)[:8]
        return bull_obs + bear_obs

    def _measure_impulse(self, df: pd.DataFrame, start: int, direction: int, window: int = 5) -> float:
        """Measure the strength of a directional move starting at `start`."""
        end = min(start + window, len(df))
        if end <= start:
            return 0.0
        ref = df["close"].iloc[start]
        if direction == 1:
            peak = df["high"].iloc[start:end].max()
            return (peak - ref) / (ref + 1e-9)
        else:
            trough = df["low"].iloc[start:end].min()
            return (ref - trough) / (ref + 1e-9)

    # ── Liquidity Sweeps ──────────────────────────────────────────────────

    def _detect_sweeps(
        self, df: pd.DataFrame,
        swing_highs: list, swing_lows: list
    ) -> List[LiquiditySweep]:
        sweeps  = []
        tol     = self.sweep_tolerance

        # Buy-side sweeps (above swing highs)
        for (sh_idx, sh_price) in swing_highs:
            for i in range(sh_idx + 1, min(sh_idx + 50, len(df))):
                bar_high  = df["high"].iloc[i]
                bar_close = df["close"].iloc[i]
                # Price wick above level then closes BELOW it
                if bar_high >= sh_price * (1 - tol) and bar_close < sh_price:
                    reversal = (bar_high - bar_close) / (bar_high + 1e-9)
                    sweeps.append(LiquiditySweep(
                        index       = i,
                        timestamp   = pd.to_datetime(df.index[i]),
                        sweep_type  = "buyside",
                        level       = sh_price,
                        sweep_high  = bar_high,
                        sweep_low   = df["low"].iloc[i],
                        reversal_size = reversal,
                        confirmed   = reversal > 0.003,
                    ))
                    break

        # Sell-side sweeps (below swing lows)
        for (sl_idx, sl_price) in swing_lows:
            for i in range(sl_idx + 1, min(sl_idx + 50, len(df))):
                bar_low   = df["low"].iloc[i]
                bar_close = df["close"].iloc[i]
                if bar_low <= sl_price * (1 + tol) and bar_close > sl_price:
                    reversal = (bar_close - bar_low) / (bar_close + 1e-9)
                    sweeps.append(LiquiditySweep(
                        index       = i,
                        timestamp   = pd.to_datetime(df.index[i]),
                        sweep_type  = "sellside",
                        level       = sl_price,
                        sweep_high  = df["high"].iloc[i],
                        sweep_low   = bar_low,
                        reversal_size = reversal,
                        confirmed   = reversal > 0.003,
                    ))
                    break

        return sorted(sweeps, key=lambda x: x.index)

    # ── Inducement ────────────────────────────────────────────────────────

    def _detect_inducement(
        self, df: pd.DataFrame,
        swing_highs: list, swing_lows: list
    ) -> List[Inducement]:
        """
        Inducement = a minor swing point that forms AFTER a structural shift
        but BEFORE price reaches the major liquidity target.
        When swept, it signals continuation toward the main target.
        """
        inds = []

        # In a bullish context: look for minor swing lows forming above a swept low
        # In a bearish context: look for minor swing highs forming below a swept high
        lb = max(3, self.swing_lookback // 2)   # smaller lookback for minor swings

        for i in range(lb, len(df) - lb):
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]
            window_h = df["high"].iloc[i - lb: i + lb + 1]
            window_l = df["low"].iloc[i - lb: i + lb + 1]

            # Minor swing high (bearish inducement)
            if h == window_h.max():
                # Is it a minor level (not a major swing high)?
                is_major = any(abs(h - msh) / (h + 1e-9) < 0.001 for (_, msh) in swing_highs)
                if not is_major:
                    # Check if a subsequent bar sweeps this level
                    swept = False
                    swept_i = None
                    for j in range(i + 1, min(i + 30, len(df))):
                        if df["high"].iloc[j] > h and df["close"].iloc[j] < h:
                            swept, swept_i = True, j
                            break
                    inds.append(Inducement(
                        index      = i,
                        timestamp  = pd.to_datetime(df.index[i]),
                        ind_type   = "bearish",
                        level      = h,
                        swept      = swept,
                        swept_index= swept_i,
                    ))

            # Minor swing low (bullish inducement)
            if l == window_l.min():
                is_major = any(abs(l - msl) / (l + 1e-9) < 0.001 for (_, msl) in swing_lows)
                if not is_major:
                    swept = False
                    swept_i = None
                    for j in range(i + 1, min(i + 30, len(df))):
                        if df["low"].iloc[j] < l and df["close"].iloc[j] > l:
                            swept, swept_i = True, j
                            break
                    inds.append(Inducement(
                        index      = i,
                        timestamp  = pd.to_datetime(df.index[i]),
                        ind_type   = "bullish",
                        level      = l,
                        swept      = swept,
                        swept_index= swept_i,
                    ))

        # Return only recently swept inducements (most actionable)
        return [ind for ind in inds if ind.swept][-20:]

    # ── Fair Value Gaps ───────────────────────────────────────────────────

    def _detect_fvg(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        FVG = 3-candle pattern where candle[1]'s body doesn't overlap
        candle[0]'s high or candle[2]'s low (bullish) vice versa (bearish).
        """
        fvgs = []
        for i in range(1, len(df) - 1):
            prev_high = df["high"].iloc[i - 1]
            prev_low  = df["low"].iloc[i - 1]
            next_high = df["high"].iloc[i + 1]
            next_low  = df["low"].iloc[i + 1]

            # Bullish FVG: gap between prev candle high and next candle low
            if next_low > prev_high:
                fvgs.append(FairValueGap(
                    index     = i,
                    timestamp = pd.to_datetime(df.index[i]),
                    fvg_type  = "bullish",
                    high      = next_low,
                    low       = prev_high,
                ))

            # Bearish FVG: gap between next candle high and prev candle low
            elif next_high < prev_low:
                fvgs.append(FairValueGap(
                    index     = i,
                    timestamp = pd.to_datetime(df.index[i]),
                    fvg_type  = "bearish",
                    high      = prev_low,
                    low       = next_high,
                ))

        return fvgs[-30:]   # keep most recent 30

    # ── OB Mitigation ─────────────────────────────────────────────────────

    def _check_ob_mitigation(self, df: pd.DataFrame, obs: List[OrderBlock]) -> None:
        """Mark an OB as mitigated once price trades back into its range."""
        latest_close = df["close"].iloc[-1]
        for ob in obs:
            if ob.ob_type == "bullish" and latest_close < ob.low:
                ob.mitigated = True
                ob.active    = False
            elif ob.ob_type == "bearish" and latest_close > ob.high:
                ob.mitigated = True
                ob.active    = False

    # ── Market Bias ───────────────────────────────────────────────────────

    def _determine_bias(
        self, df: pd.DataFrame,
        obs: List[OrderBlock], sweeps: List[LiquiditySweep]
    ) -> tuple:
        """
        Determine overall SMC bias from:
        - Recent sweep type (buyside sweep → bearish, sellside → bullish)
        - Active OB location relative to current price
        - Higher high / lower low structure
        """
        score = 0.0
        current = df["close"].iloc[-1]

        # Recent sweeps (weight: ±0.4 each, cap at 3)
        recent_sweeps = sorted(sweeps, key=lambda x: x.index)[-5:]
        for sw in recent_sweeps:
            if sw.confirmed:
                if sw.sweep_type == "sellside":   # swept lows → bullish
                    score += 0.4
                else:                             # swept highs → bearish
                    score -= 0.4

        # Active OBs relative to current price
        for ob in obs:
            if not ob.active:
                continue
            if ob.ob_type == "bullish" and ob.low < current < ob.high:
                score += 0.3   # price inside bullish OB
            if ob.ob_type == "bearish" and ob.low < current < ob.high:
                score -= 0.3

        # HH/LL structure over last 20 bars
        if len(df) >= 20:
            last_20_high = df["high"].iloc[-20:].max()
            last_20_low  = df["low"].iloc[-20:].min()
            mid_high     = df["high"].iloc[-40:-20].max() if len(df) >= 40 else last_20_high
            mid_low      = df["low"].iloc[-40:-20].min()  if len(df) >= 40 else last_20_low
            if last_20_high > mid_high:
                score += 0.2   # higher high
            if last_20_low < mid_low:
                score -= 0.2   # lower low

        # Normalise to confidence
        confidence = min(abs(score) / 1.5, 1.0)
        if score > 0.2:
            return "bullish", confidence
        if score < -0.2:
            return "bearish", confidence
        return "neutral", 1 - confidence

    # ── Signal generation ─────────────────────────────────────────────────

    def get_smc_signal(self, df: pd.DataFrame) -> dict:
        """
        High-level signal: combines SMC bias into BUY / SELL / HOLD.
        Integrate with EnsembleSignalEngine.
        """
        result  = self.analyze(df)
        current = df["close"].iloc[-1]
        atr     = df["atr"].iloc[-1] if "atr" in df.columns else current * 0.001

        signal  = {"signal": 1, "signal_name": "HOLD", "confidence": 0.5,
                   "smc_bias": result.bias, "smc_confidence": result.bias_confidence}

        # Recent confirmed sweep → entry
        recent_sweeps = [s for s in result.liquidity_sweeps if s.confirmed]
        if recent_sweeps:
            latest_sweep = max(recent_sweeps, key=lambda x: x.index)
            bars_ago = len(df) - 1 - latest_sweep.index

            if bars_ago <= 5:
                if latest_sweep.sweep_type == "sellside":   # swept lows → long
                    # Confirm: price should be near a bullish OB
                    near_ob = any(
                        ob.ob_type == "bullish" and ob.active and
                        abs(current - ob.midpoint) < atr * 2
                        for ob in result.order_blocks
                    )
                    if near_ob or result.bias == "bullish":
                        signal.update({"signal": 2, "signal_name": "BUY",
                                       "confidence": min(0.5 + result.bias_confidence * 0.5, 0.95)})
                elif latest_sweep.sweep_type == "buyside":  # swept highs → short
                    near_ob = any(
                        ob.ob_type == "bearish" and ob.active and
                        abs(current - ob.midpoint) < atr * 2
                        for ob in result.order_blocks
                    )
                    if near_ob or result.bias == "bearish":
                        signal.update({"signal": 0, "signal_name": "SELL",
                                       "confidence": min(0.5 + result.bias_confidence * 0.5, 0.95)})

        return signal
