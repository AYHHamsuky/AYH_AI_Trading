"""
Ensemble signal engine — combines XGBoost + LSTM with multi-timeframe filter.
Returns a final TradeSignal with confidence and metadata.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import numpy as np
from config import ENSEMBLE_CONFIG as EC, TIMEFRAME_MAP

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol:       str
    timeframe:    str
    timestamp:    datetime
    direction:    int           # 0=SELL, 1=HOLD, 2=BUY
    direction_name: str         # "SELL" | "HOLD" | "BUY"
    confidence:   float         # 0.0 – 1.0
    xgb_signal:   int
    xgb_confidence: float
    lstm_signal:  int
    lstm_confidence: float
    mtf_confirm:  bool          # higher-TF agrees
    atr:          float = 0.0   # for SL/TP
    close:        float = 0.0
    meta:         dict  = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return (
            self.direction != 1                         # not HOLD
            and self.confidence >= EC["min_confidence"]
            and self.mtf_confirm
            and self._in_active_session()
        )

    def _in_active_session(self) -> bool:
        if not EC["session_filter"]:
            return True
        h = self.timestamp.hour
        sess = EC["sessions"]
        return (
            (sess["london"][0] <= h < sess["london"][1]) or
            (sess["newyork"][0] <= h < sess["newyork"][1])
        )


class EnsembleSignalEngine:
    """
    Combines XGBoost + LSTM predictions with optional multi-timeframe confirmation.
    """

    _MAX_HISTORY = 1_000   # cap memory growth in long-running sessions

    def __init__(self):
        self._signal_history: list[TradeSignal] = []

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,                        # primary TF feature frame
        df_higher: Optional[pd.DataFrame] = None, # higher TF frame
        xgb_model=None,
        lstm_model=None,
    ) -> TradeSignal:
        """
        Run both models and fuse their signals.
        Returns a TradeSignal with all metadata.
        """
        # ── XGBoost signal ────────────────────────────────────────────────
        xgb_result = {"signal": 1, "confidence": 0.5}
        if xgb_model is not None:
            try:
                xgb_result = xgb_model.predict_latest(df)
            except Exception as exc:
                logger.warning("XGBoost predict failed: %s", exc)

        # ── LSTM signal ───────────────────────────────────────────────────
        lstm_result = {"signal": 1, "confidence": 0.5}
        if lstm_model is not None:
            try:
                lstm_result = lstm_model.predict_signal(df)
            except Exception as exc:
                logger.warning("LSTM predict failed: %s", exc)

        # ── Ensemble fusion (weighted vote) ───────────────────────────────
        w_xgb  = EC["xgb_weight"]
        w_lstm = EC["lstm_weight"]

        # Build probability vectors: [P_sell, P_hold, P_buy]
        xgb_probs  = self._signal_to_probs(xgb_result,  xgb_result["confidence"])
        lstm_probs = self._signal_to_probs(lstm_result, lstm_result["confidence"])

        combined_probs = w_xgb * xgb_probs + w_lstm * lstm_probs
        combined_probs /= combined_probs.sum()   # renormalise

        direction    = int(np.argmax(combined_probs))
        confidence   = float(combined_probs[direction])

        # ── Multi-timeframe confirmation ──────────────────────────────────
        mtf_confirm  = True
        if EC["mtf_confirm"] and df_higher is not None and xgb_model is not None:
            try:
                higher_result = xgb_model.predict_latest(df_higher)
                # Higher TF must agree with primary direction (or be HOLD)
                mtf_dir = higher_result["signal"]
                if mtf_dir != 1 and mtf_dir != direction:
                    mtf_confirm = False
                    # Penalise confidence if disagreement
                    confidence *= (1 - EC["mtf_weight"])
                    logger.debug("MTF disagrees: primary=%d htf=%d", direction, mtf_dir)
            except Exception as exc:
                logger.warning("MTF prediction failed: %s", exc)

        # ── Extract ATR and close from latest bar ─────────────────────────
        latest = df.iloc[-1]
        atr    = float(latest.get("atr", 0.0))
        close  = float(latest["close"])

        signal = TradeSignal(
            symbol        = symbol,
            timeframe     = timeframe,
            timestamp     = pd.to_datetime(df.index[-1]),
            direction     = direction,
            direction_name= ["SELL","HOLD","BUY"][direction],
            confidence    = confidence,
            xgb_signal    = xgb_result["signal"],
            xgb_confidence= xgb_result["confidence"],
            lstm_signal   = lstm_result["signal"],
            lstm_confidence= lstm_result["confidence"],
            mtf_confirm   = mtf_confirm,
            atr           = atr,
            close         = close,
            meta = {
                "prob_sell": combined_probs[0],
                "prob_hold": combined_probs[1],
                "prob_buy":  combined_probs[2],
                "xgb_probs": xgb_probs.tolist(),
                "lstm_probs": lstm_probs.tolist(),
            }
        )
        self._signal_history.append(signal)
        if len(self._signal_history) > self._MAX_HISTORY:
            self._signal_history = self._signal_history[-self._MAX_HISTORY:]
        self._log_signal(signal)
        return signal

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _signal_to_probs(result: dict, confidence: float) -> np.ndarray:
        """Convert a {signal, confidence} dict to a 3-element probability array."""
        probs = np.array([0.05, 0.05, 0.05])   # base probability
        signal = result.get("signal", 1)
        probs[signal] += confidence
        # Distribute remaining probability to HOLD
        probs[1] += max(0.0, 1.0 - confidence - 0.15)
        total = probs.sum()
        return probs / total

    def _log_signal(self, s: TradeSignal) -> None:
        status = "✓ ACTIONABLE" if s.is_actionable else "— skipped"
        logger.info(
            "[%s] %s %s | %s conf=%.2f xgb=%s lstm=%s mtf=%s %s",
            s.timestamp.strftime("%H:%M"),
            s.symbol, s.timeframe,
            s.direction_name, s.confidence,
            ["SELL","HOLD","BUY"][s.xgb_signal],
            ["SELL","HOLD","BUY"][s.lstm_signal],
            "✓" if s.mtf_confirm else "✗",
            status,
        )

    def recent_signals(self, n: int = 20) -> list:
        return self._signal_history[-n:]

    def signal_accuracy(self) -> Optional[float]:
        """Simple accuracy if future_return was attached."""
        correct = [
            s for s in self._signal_history
            if "actual_return" in s.meta
            and (
                (s.direction == 2 and s.meta["actual_return"] > 0) or
                (s.direction == 0 and s.meta["actual_return"] < 0)
            )
        ]
        total = [s for s in self._signal_history if "actual_return" in s.meta]
        if not total:
            return None
        return len(correct) / len(total)
