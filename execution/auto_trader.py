"""
Auto-Execution Engine
─────────────────────
Watches signal pipeline continuously. When ALL quality gates pass,
it automatically places the trade — paper or live.

Quality gates (ALL must pass):
  1. Ensemble confidence ≥ threshold
  2. SMC direction agrees
  3. Multi-timeframe confirmation
  4. Signal persisted for N bars
  5. Cooldown between same-symbol trades
  6. Not in a restricted session
  7. Kill switch not triggered
  8. Max daily trades not exceeded
"""

import json
import logging
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Callable

logger = logging.getLogger(__name__)


@dataclass
class ExecutionLog:
    timestamp:    str
    symbol:       str
    timeframe:    str
    direction:    str          # BUY | SELL
    mode:         str          # paper | live
    lot_size:     float
    entry_price:  float
    sl:           float
    tp:           float
    confidence:   float
    smc_bias:     str
    xgb_conf:     float
    lstm_conf:    float
    trade_id:     Optional[str] = None
    status:       str = "pending"   # pending | filled | failed | cancelled
    pnl:          Optional[float] = None
    close_time:   Optional[str]  = None
    close_reason: Optional[str]  = None
    notes:        str = ""


@dataclass
class SignalState:
    """Tracks signal persistence per symbol+timeframe."""
    symbol:      str
    timeframe:   str
    direction:   int      # 0=SELL, 1=HOLD, 2=BUY
    bars_held:   int = 0
    first_seen:  Optional[datetime] = None
    last_conf:   float = 0.0


class AutoTrader:
    """
    Auto-execution engine. Attach to your main trading loop.
    
    Usage:
        trader = AutoTrader(broker=oanda_broker, mode="paper")
        trader.start()
        # In your tick loop:
        trader.on_signal(signal, smc_signal, risk_order)
    """

    def __init__(
        self,
        broker,                      # OANDABroker | MT5Broker instance
        risk_manager,                # RiskManager instance
        notifier=None,               # TelegramNotifier instance
        mode: str = "paper",         # "paper" | "live"
        config: dict = None,
    ):
        from config import AUTO_TRADER_CONFIG
        self.cfg          = config or AUTO_TRADER_CONFIG
        self.broker       = broker
        self.risk_manager = risk_manager
        self.notifier     = notifier
        self.mode         = mode

        # State
        self._enabled       = False
        self._killed        = False
        self._signal_states: Dict[str, SignalState] = {}
        self._last_trade_bar: Dict[str, int]         = defaultdict(lambda: -999)
        self._trades_today: List[ExecutionLog]        = []
        self._all_trades:   List[ExecutionLog]        = []
        self._daily_reset_date = date.today()
        self._bar_counter   = 0
        self._paper_balance = self.cfg.get("paper_initial_balance", 10_000)
        self._paper_peak    = self._paper_balance
        self._lock          = threading.Lock()

        # Load paper trade history if exists
        self._log_file = self.cfg.get("paper_log_file", "logs/paper_trades.jsonl")
        self._load_log()

        # External callbacks (optional)
        self.on_trade_opened: Optional[Callable] = None
        self.on_trade_closed: Optional[Callable] = None
        self.on_kill_switch:  Optional[Callable] = None

    # ── Control ───────────────────────────────────────────────────────────
    def enable(self) -> None:
        self._enabled = True
        logger.info("AutoTrader ENABLED [%s mode]", self.mode.upper())
        if self.notifier:
            self.notifier.info(f"🤖 AutoTrader ENABLED — {self.mode.upper()} mode")

    def disable(self) -> None:
        self._enabled = False
        logger.info("AutoTrader DISABLED")
        if self.notifier:
            self.notifier.info("🤖 AutoTrader DISABLED")

    def reset_kill_switch(self) -> None:
        self._killed = False
        self.risk_manager.reset_kill_switch()
        logger.warning("Kill switch reset by operator")

    @property
    def is_active(self) -> bool:
        return self._enabled and not self._killed

    # ── Main signal handler ───────────────────────────────────────────────
    def on_signal(
        self,
        signal,           # TradeSignal from EnsembleSignalEngine
        smc_signal: dict, # from SMCEngine.get_smc_signal()
        df,               # current feature DataFrame (for trailing stop updates)
    ) -> Optional[ExecutionLog]:
        """
        Call this every bar with the latest signal.
        Returns ExecutionLog if a trade was placed, else None.
        """
        self._bar_counter += 1
        self._reset_daily_if_needed()

        if not self.is_active:
            return None

        symbol    = signal.symbol
        timeframe = signal.timeframe
        key       = f"{symbol}_{timeframe}"

        # ── Update trailing stops on existing positions ────────────────────
        self._update_trailing_stops(df)

        # ── Gate 1: Risk kill switch ──────────────────────────────────────
        risk_stats = self.risk_manager.stats()
        if risk_stats["kill_switch"]:
            if not self._killed:
                self._killed = True
                logger.critical("KILL SWITCH: AutoTrader suspended")
                if self.notifier:
                    self.notifier.kill_switch_triggered("Risk limits breached", risk_stats["balance"])
                if self.on_kill_switch:
                    self.on_kill_switch(risk_stats)
            return None

        # ── Gate 2: Signal must be BUY or SELL ───────────────────────────
        if signal.direction == 1:   # HOLD
            self._decay_signal_state(key)
            return None

        # ── Gate 3: Ensemble confidence ───────────────────────────────────
        if signal.confidence < self.cfg["min_ensemble_conf"]:
            logger.debug("[%s] Ensemble conf %.2f < %.2f — skip",
                         key, signal.confidence, self.cfg["min_ensemble_conf"])
            self._decay_signal_state(key)
            return None

        # ── Gate 4: SMC confirmation ──────────────────────────────────────
        if self.cfg["require_smc_confirm"]:
            smc_dir  = smc_signal.get("signal", 1)
            smc_conf = smc_signal.get("confidence", 0.0)
            if smc_dir != signal.direction:
                logger.debug("[%s] SMC disagrees (ens=%d smc=%d) — skip",
                             key, signal.direction, smc_dir)
                self._decay_signal_state(key)
                return None
            if smc_conf < self.cfg["min_smc_conf"]:
                logger.debug("[%s] SMC conf %.2f < %.2f — skip",
                             key, smc_conf, self.cfg["min_smc_conf"])
                return None

        # ── Gate 5: MTF confirmation ──────────────────────────────────────
        if self.cfg["require_mtf_confirm"] and not signal.mtf_confirm:
            logger.debug("[%s] MTF not confirmed — skip", key)
            return None

        # ── Gate 6: Signal persistence ─────────────────────────────────────
        state = self._update_signal_state(key, signal)
        if state.bars_held < self.cfg["signal_persistence"]:
            logger.debug("[%s] Signal held %d/%d bars — waiting",
                         key, state.bars_held, self.cfg["signal_persistence"])
            return None

        # ── Gate 7: Cooldown ──────────────────────────────────────────────
        bars_since_last = self._bar_counter - self._last_trade_bar[key]
        if bars_since_last < self.cfg["cooldown_bars"]:
            logger.debug("[%s] Cooldown active (%d/%d bars) — skip",
                         key, bars_since_last, self.cfg["cooldown_bars"])
            return None

        # ── Gate 8: Daily trade limit ─────────────────────────────────────
        if len(self._trades_today) >= self.cfg["max_trades_per_day"]:
            logger.debug("Daily trade limit reached (%d)", self.cfg["max_trades_per_day"])
            return None

        symbol_today = sum(1 for t in self._trades_today if t.symbol == symbol)
        if symbol_today >= self.cfg["max_trades_per_symbol"]:
            logger.debug("[%s] Symbol daily limit reached", symbol)
            return None

        # ── Gate 9: Session filter ────────────────────────────────────────
        if not self._in_valid_session():
            logger.debug("[%s] Outside valid session — skip", key)
            return None

        # ── ALL GATES PASSED — approve and execute ─────────────────────────
        risk_order = self.risk_manager.approve_trade(
            symbol     = symbol,
            direction  = signal.direction,
            entry      = signal.close,
            atr        = signal.atr,
            confidence = signal.confidence,
        )
        if risk_order is None:
            logger.debug("[%s] Risk manager rejected trade", key)
            return None

        log = self._execute(signal, risk_order, smc_signal)
        if log:
            self._last_trade_bar[key] = self._bar_counter
            self._reset_signal_state(key)
        return log

    # ── Execution ─────────────────────────────────────────────────────────
    def _execute(self, signal, order, smc_signal: dict) -> Optional[ExecutionLog]:
        direction_name = ["SELL", "HOLD", "BUY"][signal.direction]
        logger.info(
            "AUTO-EXECUTE: %s %s lot=%.2f entry=%.5f SL=%.5f TP=%.5f mode=%s",
            direction_name, signal.symbol, order.lot_size,
            order.entry_price, order.stop_loss, order.take_profit, self.mode
        )

        log = ExecutionLog(
            timestamp   = datetime.utcnow().isoformat(),
            symbol      = signal.symbol,
            timeframe   = signal.timeframe,
            direction   = direction_name,
            mode        = self.mode,
            lot_size    = order.lot_size,
            entry_price = order.entry_price,
            sl          = order.stop_loss,
            tp          = order.take_profit,
            confidence  = signal.confidence,
            smc_bias    = smc_signal.get("smc_bias", "unknown"),
            xgb_conf    = signal.xgb_confidence,
            lstm_conf   = signal.lstm_confidence,
        )

        if self.mode == "paper":
            log.trade_id = f"PAPER_{datetime.utcnow().strftime('%H%M%S%f')}"
            log.status   = "filled"
            logger.info("📄 PAPER TRADE: %s", log.trade_id)
        else:
            # Live execution
            try:
                units = self.broker.lot_to_units(signal.symbol, order.lot_size) \
                        if hasattr(self.broker, "lot_to_units") else int(order.lot_size * 100_000)
                trade_id = self.broker.place_market_order(
                    symbol    = signal.symbol,
                    direction = signal.direction,
                    units     = units,
                    sl        = order.stop_loss,
                    tp        = order.take_profit,
                    comment   = f"AI bot conf={signal.confidence:.2f}",
                )
                if trade_id is None:
                    log.status = "failed"
                    log.notes  = "Broker rejected order"
                    logger.error("Live order failed for %s", signal.symbol)
                    self._save_log(log)
                    return None
                log.trade_id = trade_id
                log.status   = "filled"
            except Exception as exc:
                log.status = "failed"
                log.notes  = str(exc)
                logger.exception("Live order exception: %s", exc)
                if self.notifier:
                    self.notifier.error_alert(f"Order failed: {signal.symbol}\n{exc}")
                self._save_log(log)
                return None

        # Register with risk manager
        self.risk_manager.register_open(order)
        self._trades_today.append(log)
        self._all_trades.append(log)
        self._save_log(log)

        # Notify
        if self.notifier:
            self.notifier.trade_opened(
                symbol    = signal.symbol,
                direction = direction_name,
                lot       = order.lot_size,
                entry     = order.entry_price,
                sl        = order.stop_loss,
                tp        = order.take_profit,
                rr        = order.rr_ratio,
                confidence= signal.confidence,
            )

        if self.on_trade_opened:
            self.on_trade_opened(log)

        return log

    # ── Trailing stops ────────────────────────────────────────────────────
    def _update_trailing_stops(self, df) -> None:
        if self.mode == "paper":
            return
        try:
            positions = self.broker.get_open_trades()
        except Exception:
            return
        for pos in positions:
            symbol = pos["symbol"]
            price  = pos.get("entry_price", 0.0)
            try:
                p = self.broker.get_price(symbol)
                if p:
                    price = p["mid"]
            except Exception:
                pass
            atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else price * 0.001
            from risk.manager import TradeOrder
            lot = pos.get("lot_size") or pos.get("units", 0.01)
            dummy = TradeOrder(
                symbol      = symbol,
                direction   = 2 if pos["direction"] == "BUY" else 0,
                lot_size    = float(lot),
                entry_price = float(pos.get("entry_price", price)),
                stop_loss   = float(pos.get("sl", 0)),
                take_profit = float(pos.get("tp", 0)),
                rr_ratio    = 1.5,
                risk_amount = 0,
            )
            new_sl = self.risk_manager.trailing_sl(dummy, price, atr)
            cur_sl = float(pos.get("sl") or 0)
            if new_sl and cur_sl and abs(new_sl - cur_sl) > atr * 0.1:
                if hasattr(self.broker, "modify_trade_sl"):
                    self.broker.modify_trade_sl(pos["id"], new_sl)
                elif hasattr(self.broker, "modify_sl"):
                    self.broker.modify_sl(int(pos["id"]), new_sl)

    # ── Signal state tracking ─────────────────────────────────────────────
    def _update_signal_state(self, key: str, signal) -> SignalState:
        state = self._signal_states.get(key)
        if state is None or state.direction != signal.direction:
            state = SignalState(
                symbol     = signal.symbol,
                timeframe  = signal.timeframe,
                direction  = signal.direction,
                bars_held  = 1,
                first_seen = datetime.utcnow(),
                last_conf  = signal.confidence,
            )
        else:
            state.bars_held += 1
            state.last_conf  = signal.confidence
        self._signal_states[key] = state
        return state

    def _decay_signal_state(self, key: str) -> None:
        if key in self._signal_states:
            self._signal_states[key].bars_held = max(0, self._signal_states[key].bars_held - 1)

    def _reset_signal_state(self, key: str) -> None:
        self._signal_states.pop(key, None)

    # ── Session filter ────────────────────────────────────────────────────
    def _in_valid_session(self) -> bool:
        h = datetime.utcnow().hour
        london   = self.cfg.get("trade_london",  True)  and (7  <= h < 16)
        newyork  = self.cfg.get("trade_newyork", True)  and (12 <= h < 21)
        tokyo    = self.cfg.get("trade_tokyo",   False) and (0  <= h < 8)
        return london or newyork or tokyo

    # ── Daily reset ───────────────────────────────────────────────────────
    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._daily_reset_date:
            self._trades_today     = []
            self._daily_reset_date = today
            logger.info("Daily trade counters reset")

    # ── Stats ─────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        closed = [t for t in self._all_trades if t.pnl is not None]
        wins   = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        return {
            "mode":            self.mode,
            "enabled":         self._enabled,
            "killed":          self._killed,
            "trades_today":    len(self._trades_today),
            "total_trades":    len(self._all_trades),
            "open_trades":     len([t for t in self._all_trades if t.status=="filled" and t.pnl is None]),
            "closed_trades":   len(closed),
            "win_rate":        len(wins) / (len(closed) + 1e-9),
            "total_pnl":       sum(t.pnl for t in closed),
            "avg_win":         sum(t.pnl for t in wins)   / (len(wins)   + 1e-9),
            "avg_loss":        sum(t.pnl for t in losses) / (len(losses) + 1e-9),
            "paper_balance":   self._paper_balance,
        }

    def recent_trades(self, n: int = 20) -> List[ExecutionLog]:
        return list(reversed(self._all_trades[-n:]))

    # ── Persistence ───────────────────────────────────────────────────────
    def _save_log(self, log: ExecutionLog) -> None:
        """Append a single trade record as a JSON line (O(1), no full-file rewrite)."""
        try:
            os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
            with open(self._log_file, "a") as f:
                f.write(json.dumps(asdict(log), default=str) + "\n")
        except Exception as exc:
            logger.warning("Log save failed: %s", exc)

    def _load_log(self) -> None:
        if not os.path.exists(self._log_file):
            return
        try:
            with open(self._log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._all_trades.append(ExecutionLog(**json.loads(line)))
                    except Exception:
                        pass
            logger.info("Loaded %d historical trades from log", len(self._all_trades))
        except Exception as exc:
            logger.warning("Log load failed: %s", exc)
