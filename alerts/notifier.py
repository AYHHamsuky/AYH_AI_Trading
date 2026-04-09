"""
Telegram notifier — sends trade alerts, daily reports, kill-switch warnings.
Supports multi-subscriber broadcast: users can /subscribe and /unsubscribe.

Features:
  • Rich signal messages with entry/SL/TP levels and R:R
  • Inline keyboard buttons for quick actions
  • Rate-limited message queue (respects Telegram 30 msg/sec limit)
  • Chart image sending via sendPhoto (matplotlib candlestick)
  • Edit-in-place: updates original signal when trade opens/closes
  • Extended bot commands: /performance, /openpositions, /settings, etc.
"""

import io
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Set, Dict
import requests
from config import TELEGRAM_CONFIG as TC, RISK_CONFIG

logger = logging.getLogger(__name__)

EMOJI = {
    "buy":     "📈",
    "sell":    "📉",
    "hold":    "⏸️",
    "tp":      "✅",
    "sl":      "🛑",
    "kill":    "🚨",
    "report":  "📊",
    "error":   "❌",
    "info":    "ℹ️",
    "signal":  "🔔",
    "bull":    "🟢",
    "bear":    "🔴",
    "neutral": "⚪",
    "chart":   "📉",
    "money":   "💰",
    "clock":   "🕐",
    "star":    "⭐",
    "fire":    "🔥",
    "shield":  "🛡️",
}

SUBSCRIBERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "subscribers.json"
)

# Rate limit: Telegram allows ~30 msgs/sec, we stay under at 25/sec
_RATE_LIMIT = 25
_RATE_WINDOW = 1.0  # seconds


class TelegramNotifier:
    """
    Telegram bot notifier with multi-subscriber broadcast support,
    rate-limited queue, inline keyboards, chart images, and edit-in-place tracking.
    """

    def __init__(self):
        self.token    = TC["token"]
        self.chat_id  = TC["chat_id"]
        self._base    = f"https://api.telegram.org/bot{self.token}"
        self._enabled = bool(self.token and self.chat_id)

        # Multi-subscriber state
        self._subscribers: Set[str] = set()
        self._load_subscribers()

        # Polling state
        self._poll_offset   = 0
        self._poll_thread   = None
        self._poll_running  = False

        # Rate-limited message queue
        self._msg_queue: queue.Queue = queue.Queue()
        self._send_times: deque = deque()
        self._queue_thread: Optional[threading.Thread] = None
        self._queue_running = False

        # Edit-in-place: track signal message IDs per symbol
        # key = "SYMBOL_TF" → {chat_id: message_id}
        self._signal_msg_ids: Dict[str, Dict[str, int]] = {}

        # External data providers (set by main.py)
        self._auto_trader = None
        self._risk_manager = None
        self._broker = None

    def set_context(self, auto_trader=None, risk_manager=None, broker=None):
        """Attach external objects for /performance, /openpositions etc."""
        self._auto_trader = auto_trader
        self._risk_manager = risk_manager
        self._broker = broker

    # ── Rate-limited queue ────────────────────────────────────────────────
    def _start_queue(self) -> None:
        if self._queue_thread and self._queue_thread.is_alive():
            return
        self._queue_running = True
        self._queue_thread = threading.Thread(
            target=self._queue_loop, daemon=True, name="tg-queue"
        )
        self._queue_thread.start()

    def _queue_loop(self) -> None:
        while self._queue_running:
            try:
                item = self._msg_queue.get(timeout=1)
            except queue.Empty:
                continue
            self._wait_for_rate_limit()
            try:
                item["func"](**item["kwargs"])
            except Exception as exc:
                logger.warning("Queue send error: %s", exc)
            self._send_times.append(time.monotonic())

    def _wait_for_rate_limit(self) -> None:
        now = time.monotonic()
        while self._send_times and (now - self._send_times[0]) > _RATE_WINDOW:
            self._send_times.popleft()
        if len(self._send_times) >= _RATE_LIMIT:
            sleep_for = _RATE_WINDOW - (now - self._send_times[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ── Core send ─────────────────────────────────────────────────────────
    def send(self, message: str, chat_id: str = None, parse_mode: str = "HTML",
             reply_markup: dict = None) -> Optional[int]:
        """Send a text message. Returns message_id on success."""
        if not self._enabled:
            logger.debug("Telegram disabled (no token/chat_id)")
            return None
        target = str(chat_id) if chat_id else str(self.chat_id)
        payload = {
            "chat_id": target,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)
        try:
            resp = requests.post(
                f"{self._base}/sendMessage",
                json=payload,
                timeout=10,
            )
            if not resp.ok:
                logger.warning("Telegram send failed (chat=%s): %s", target, resp.text)
                return None
            return resp.json().get("result", {}).get("message_id")
        except Exception as exc:
            logger.warning("Telegram error: %s", exc)
            return None

    def edit_message(self, chat_id: str, message_id: int, text: str,
                     parse_mode: str = "HTML", reply_markup: dict = None) -> bool:
        """Edit an existing message in-place."""
        if not self._enabled:
            return False
        payload = {
            "chat_id": str(chat_id),
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)
        try:
            resp = requests.post(
                f"{self._base}/editMessageText",
                json=payload,
                timeout=10,
            )
            return resp.ok
        except Exception as exc:
            logger.warning("Edit message error: %s", exc)
            return False

    def send_photo(self, photo_bytes: bytes, caption: str = "",
                   chat_id: str = None, parse_mode: str = "HTML") -> Optional[int]:
        """Send a photo (bytes). Returns message_id on success."""
        if not self._enabled:
            return None
        target = str(chat_id) if chat_id else str(self.chat_id)
        try:
            resp = requests.post(
                f"{self._base}/sendPhoto",
                data={"chat_id": target, "caption": caption, "parse_mode": parse_mode},
                files={"photo": ("chart.png", photo_bytes, "image/png")},
                timeout=15,
            )
            if not resp.ok:
                logger.warning("Telegram sendPhoto failed: %s", resp.text)
                return None
            return resp.json().get("result", {}).get("message_id")
        except Exception as exc:
            logger.warning("Telegram photo error: %s", exc)
            return None

    def answer_callback(self, callback_query_id: str, text: str = "") -> None:
        """Answer an inline-button callback query."""
        if not self._enabled:
            return
        try:
            requests.post(
                f"{self._base}/answerCallbackQuery",
                json={"callback_query_id": callback_query_id, "text": text},
                timeout=5,
            )
        except Exception:
            pass

    def broadcast(self, message: str, parse_mode: str = "HTML",
                  reply_markup: dict = None) -> Dict[str, int]:
        """Send message to owner + all subscribers. Returns {chat_id: msg_id}."""
        self._start_queue()
        recipients = {str(self.chat_id)} | self._subscribers
        msg_ids = {}
        for cid in recipients:
            if cid:
                mid = self.send(message, chat_id=cid, parse_mode=parse_mode,
                                reply_markup=reply_markup)
                if mid:
                    msg_ids[cid] = mid
        return msg_ids

    def broadcast_photo(self, photo_bytes: bytes, caption: str = "",
                        parse_mode: str = "HTML") -> Dict[str, int]:
        """Send photo to owner + all subscribers."""
        recipients = {str(self.chat_id)} | self._subscribers
        msg_ids = {}
        for cid in recipients:
            if cid:
                self._wait_for_rate_limit()
                mid = self.send_photo(photo_bytes, caption=caption,
                                      chat_id=cid, parse_mode=parse_mode)
                self._send_times.append(time.monotonic())
                if mid:
                    msg_ids[cid] = mid
        return msg_ids

    # ── Subscriber management ─────────────────────────────────────────────
    def add_subscriber(self, chat_id: str) -> bool:
        chat_id = str(chat_id)
        if chat_id in self._subscribers:
            return False
        self._subscribers.add(chat_id)
        self._save_subscribers()
        logger.info("New subscriber: %s (total=%d)", chat_id, len(self._subscribers))
        return True

    def remove_subscriber(self, chat_id: str) -> bool:
        chat_id = str(chat_id)
        if chat_id not in self._subscribers:
            return False
        self._subscribers.discard(chat_id)
        self._save_subscribers()
        logger.info("Unsubscribed: %s (total=%d)", chat_id, len(self._subscribers))
        return True

    def subscriber_count(self) -> int:
        return len(self._subscribers) + (1 if self.chat_id else 0)

    def _load_subscribers(self) -> None:
        try:
            os.makedirs(os.path.dirname(SUBSCRIBERS_FILE), exist_ok=True)
            if os.path.exists(SUBSCRIBERS_FILE):
                with open(SUBSCRIBERS_FILE) as f:
                    data = json.load(f)
                self._subscribers = set(str(x) for x in data.get("subscribers", []))
                self._poll_offset = data.get("poll_offset", 0)
                logger.info("Loaded %d subscribers", len(self._subscribers))
        except Exception as exc:
            logger.warning("Could not load subscribers: %s", exc)

    def _save_subscribers(self) -> None:
        try:
            os.makedirs(os.path.dirname(SUBSCRIBERS_FILE), exist_ok=True)
            with open(SUBSCRIBERS_FILE, "w") as f:
                json.dump({
                    "subscribers": list(self._subscribers),
                    "poll_offset": self._poll_offset,
                }, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save subscribers: %s", exc)

    # ── Chart generation ──────────────────────────────────────────────────
    def _generate_chart(self, df, symbol: str, timeframe: str,
                        entry: float = None, sl: float = None,
                        tp: float = None) -> Optional[bytes]:
        """Generate a candlestick chart image with optional levels."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import FancyArrowPatch
            import numpy as np

            # Use last 60 bars for a clean chart
            chart_df = df.tail(60).copy()
            if chart_df.empty:
                return None

            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a1a2e")
            ax.set_facecolor("#1a1a2e")

            # Draw candlesticks
            x = range(len(chart_df))
            opens  = chart_df["open"].values
            highs  = chart_df["high"].values
            lows   = chart_df["low"].values
            closes = chart_df["close"].values

            colors = ["#26a69a" if c >= o else "#ef5350"
                      for o, c in zip(opens, closes)]

            for i in range(len(chart_df)):
                ax.plot([x[i], x[i]], [lows[i], highs[i]],
                        color=colors[i], linewidth=0.8)
                ax.plot([x[i], x[i]],
                        [min(opens[i], closes[i]), max(opens[i], closes[i])],
                        color=colors[i], linewidth=3.5)

            # Draw entry/SL/TP horizontal lines
            last_x = len(chart_df) - 1
            if entry is not None:
                ax.axhline(y=entry, color="#FFD700", linestyle="--",
                           linewidth=1.2, alpha=0.9, label=f"Entry: {entry:.5f}")
            if sl is not None:
                ax.axhline(y=sl, color="#ef5350", linestyle="--",
                           linewidth=1.2, alpha=0.9, label=f"SL: {sl:.5f}")
            if tp is not None:
                ax.axhline(y=tp, color="#26a69a", linestyle="--",
                           linewidth=1.2, alpha=0.9, label=f"TP: {tp:.5f}")

            ax.set_title(f"{symbol} {timeframe}", color="white",
                         fontsize=14, fontweight="bold")
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333")

            if entry or sl or tp:
                ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
                          edgecolor="#444", labelcolor="white")

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        except Exception as exc:
            logger.warning("Chart generation failed: %s", exc)
            return None

    # ── Inline keyboard helpers ───────────────────────────────────────────
    @staticmethod
    def _signal_keyboard(symbol: str, timeframe: str, direction: str) -> dict:
        """Inline keyboard for signal messages."""
        cb_prefix = f"{symbol}_{timeframe}"
        return {
            "inline_keyboard": [
                [
                    {"text": f"{'🟢 Execute' if direction == 'BUY' else '🔴 Execute'}",
                     "callback_data": f"exec_{cb_prefix}_{direction}"},
                    {"text": "❌ Dismiss",
                     "callback_data": f"dismiss_{cb_prefix}"},
                ],
                [
                    {"text": "📊 Show Chart",
                     "callback_data": f"chart_{cb_prefix}"},
                    {"text": "📋 Details",
                     "callback_data": f"details_{cb_prefix}"},
                ],
            ]
        }

    # ── Background polling for commands & callbacks ───────────────────────
    def start_polling(self) -> None:
        """Start background thread to handle commands and inline callbacks."""
        if self._poll_thread and self._poll_thread.is_alive():
            return
        if not self._enabled:
            return
        self._poll_running = True
        self._poll_thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="tg-poll"
        )
        self._poll_thread.start()
        self._start_queue()
        logger.info("Telegram polling started (subscribers: %d)", len(self._subscribers))

    def stop_polling(self) -> None:
        self._poll_running = False
        self._queue_running = False

    def _poll_loop(self) -> None:
        while self._poll_running:
            try:
                self._poll_once()
            except Exception as exc:
                logger.debug("Poll error: %s", exc)
            time.sleep(3)

    def _poll_once(self) -> None:
        resp = requests.get(
            f"{self._base}/getUpdates",
            params={"offset": self._poll_offset + 1, "timeout": 2, "limit": 50},
            timeout=15,
        )
        if not resp.ok:
            return
        updates = resp.json().get("result", [])
        for upd in updates:
            self._poll_offset = upd["update_id"]

            # Handle inline button callbacks
            callback = upd.get("callback_query")
            if callback:
                self._handle_callback(callback)
                continue

            msg = upd.get("message") or upd.get("channel_post", {})
            if not msg:
                continue
            text    = (msg.get("text") or "").strip().lower()
            chat_id = str(msg.get("chat", {}).get("id", ""))
            if not chat_id:
                continue

            self._handle_command(text, chat_id)

        if updates:
            self._save_subscribers()

    def _handle_command(self, text: str, chat_id: str) -> None:
        """Handle text commands from users."""
        if text.startswith("/subscribe"):
            if self.add_subscriber(chat_id):
                self.send(
                    f"{EMOJI['signal']} <b>Subscribed!</b>\n"
                    "You will now receive all trading signals and alerts.\n"
                    "Send /unsubscribe to stop.",
                    chat_id=chat_id,
                )
            else:
                self.send(
                    f"{EMOJI['info']} You are already subscribed.",
                    chat_id=chat_id,
                )

        elif text.startswith("/unsubscribe"):
            if self.remove_subscriber(chat_id):
                self.send(
                    f"{EMOJI['info']} <b>Unsubscribed.</b> You will no longer receive signals.",
                    chat_id=chat_id,
                )
            else:
                self.send(
                    f"{EMOJI['info']} You are not subscribed.",
                    chat_id=chat_id,
                )

        elif text.startswith("/status"):
            self._cmd_status(chat_id)

        elif text.startswith("/performance"):
            self._cmd_performance(chat_id)

        elif text.startswith("/openpositions") or text.startswith("/positions"):
            self._cmd_open_positions(chat_id)

        elif text.startswith("/settings"):
            self._cmd_settings(chat_id)

        elif text.startswith("/balance"):
            self._cmd_balance(chat_id)

        elif text.startswith("/start") or text.startswith("/help"):
            self.send(
                f"{EMOJI['signal']} <b>AYH AI Trading Bot</b>\n\n"
                "<b>Commands:</b>\n"
                "/subscribe — receive live signals &amp; alerts\n"
                "/unsubscribe — stop receiving signals\n"
                "/status — bot status &amp; uptime\n"
                "/performance — trading performance stats\n"
                "/positions — open positions\n"
                "/balance — account balance\n"
                "/settings — current risk settings\n\n"
                "<i>Signals include SMC, XGBoost &amp; LSTM analysis\n"
                "with entry/SL/TP levels and chart images.</i>",
                chat_id=chat_id,
            )

    def _handle_callback(self, callback: dict) -> None:
        """Handle inline keyboard button presses."""
        cb_id   = callback.get("id", "")
        data    = callback.get("data", "")
        chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))

        if data.startswith("exec_"):
            # exec_SYMBOL_TF_DIRECTION
            parts = data.split("_", 3)
            if len(parts) >= 4:
                sym, tf, direction = parts[1], parts[2], parts[3]
                self.answer_callback(cb_id, f"⚡ Executing {direction} {sym}...")
                self.send(
                    f"{EMOJI['info']} Manual execution requested for "
                    f"<b>{direction} {sym} {tf}</b>\n"
                    "<i>Use AutoTrader for automated execution.</i>",
                    chat_id=chat_id,
                )
            else:
                self.answer_callback(cb_id, "Invalid callback data")

        elif data.startswith("dismiss_"):
            self.answer_callback(cb_id, "Signal dismissed")
            self.send(f"{EMOJI['info']} Signal dismissed.", chat_id=chat_id)

        elif data.startswith("chart_"):
            parts = data.split("_", 2)
            if len(parts) >= 3:
                key = f"{parts[1]}_{parts[2]}"
                self.answer_callback(cb_id, "Generating chart...")
                self.send(
                    f"{EMOJI['chart']} Chart for <b>{parts[1]} {parts[2]}</b> "
                    "will be sent with the next signal.",
                    chat_id=chat_id,
                )
            else:
                self.answer_callback(cb_id)

        elif data.startswith("details_"):
            parts = data.split("_", 2)
            if len(parts) >= 3:
                self.answer_callback(cb_id, "Loading details...")
                self._cmd_status(chat_id)
            else:
                self.answer_callback(cb_id)

        else:
            self.answer_callback(cb_id)

    # ── Extended bot commands ─────────────────────────────────────────────
    def _cmd_status(self, chat_id: str) -> None:
        lines = [f"{EMOJI['info']} <b>AYH Bot Status</b>"]
        lines.append(f"Subscribers: {self.subscriber_count()}")

        if self._risk_manager:
            stats = self._risk_manager.stats()
            lines.append(f"Balance: ${stats.get('balance', 0):,.2f}")
            lines.append(f"Daily loss: ${stats.get('daily_loss', 0):.2f}")
            lines.append(f"Drawdown: {stats.get('drawdown_pct', 0):.2f}%")
            ks = "🟢 OK" if not stats.get("kill_switch") else "🔴 TRIGGERED"
            lines.append(f"Kill switch: {ks}")

        if self._auto_trader:
            at_stats = self._auto_trader.stats()
            mode = "📄 Paper" if self._auto_trader.mode == "paper" else "💰 Live"
            lines.append(f"Mode: {mode}")
            lines.append(f"Open trades: {at_stats.get('open_trades', 0)}")

        lines.append(f"\n{EMOJI['clock']} {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        self.send("\n".join(lines), chat_id=chat_id)

    def _cmd_performance(self, chat_id: str) -> None:
        if not self._auto_trader:
            self.send(f"{EMOJI['info']} AutoTrader not active.", chat_id=chat_id)
            return

        stats = self._auto_trader.stats()
        total    = stats.get("total_trades", 0)
        wins     = stats.get("wins", 0)
        losses   = stats.get("losses", 0)
        win_rate = stats.get("win_rate", 0)
        pnl      = stats.get("total_pnl", 0)
        today_pnl = stats.get("daily_pnl", 0)
        best     = stats.get("best_trade", 0)
        worst    = stats.get("worst_trade", 0)

        pnl_sign  = "+" if pnl >= 0 else ""
        today_sign= "+" if today_pnl >= 0 else ""

        msg = (
            f"{EMOJI['report']} <b>Performance Summary</b>\n"
            f"{'─'*28}\n"
            f"Total trades : {total}\n"
            f"Wins / Losses: {wins} / {losses}\n"
            f"Win rate     : {win_rate:.1%}\n"
            f"{'─'*28}\n"
            f"Total PnL    : <b>{pnl_sign}${pnl:.2f}</b>\n"
            f"Today's PnL  : {today_sign}${today_pnl:.2f}\n"
            f"Best trade   : +${best:.2f}\n"
            f"Worst trade  : -${abs(worst):.2f}\n"
            f"{'─'*28}\n"
            f"{EMOJI['clock']} {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        )
        self.send(msg, chat_id=chat_id)

    def _cmd_open_positions(self, chat_id: str) -> None:
        if not self._auto_trader:
            self.send(f"{EMOJI['info']} AutoTrader not active.", chat_id=chat_id)
            return

        stats = self._auto_trader.stats()
        open_trades = stats.get("open_trades_list", [])

        if not open_trades:
            self.send(f"{EMOJI['info']} No open positions.", chat_id=chat_id)
            return

        lines = [f"{EMOJI['money']} <b>Open Positions</b>\n{'─'*28}"]
        for t in open_trades:
            dir_emoji = EMOJI["buy"] if t.get("direction") == "BUY" else EMOJI["sell"]
            lines.append(
                f"{dir_emoji} <b>{t.get('symbol', '?')}</b> {t.get('direction', '?')}\n"
                f"   Entry: {t.get('entry', 0):.5f}  |  Lot: {t.get('lot', 0):.2f}\n"
                f"   SL: {t.get('sl', 0):.5f}  |  TP: {t.get('tp', 0):.5f}"
            )
        lines.append(f"\n{EMOJI['clock']} {datetime.utcnow().strftime('%H:%M')} UTC")
        self.send("\n".join(lines), chat_id=chat_id)

    def _cmd_balance(self, chat_id: str) -> None:
        if self._broker:
            try:
                bal = self._broker.account_balance()
                self.send(
                    f"{EMOJI['money']} <b>Account Balance</b>\n"
                    f"${bal:,.2f}",
                    chat_id=chat_id,
                )
                return
            except Exception:
                pass
        if self._risk_manager:
            stats = self._risk_manager.stats()
            self.send(
                f"{EMOJI['money']} <b>Balance</b>: ${stats.get('balance', 0):,.2f}",
                chat_id=chat_id,
            )
        else:
            self.send(f"{EMOJI['info']} Balance not available.", chat_id=chat_id)

    def _cmd_settings(self, chat_id: str) -> None:
        msg = (
            f"{EMOJI['shield']} <b>Risk Settings</b>\n"
            f"{'─'*28}\n"
            f"Risk per trade : {RISK_CONFIG['account_risk_pct']}%\n"
            f"Max open trades: {RISK_CONFIG['max_open_trades']}\n"
            f"SL (ATR mult)  : {RISK_CONFIG['sl_atr_mult']}×\n"
            f"TP (ATR mult)  : {RISK_CONFIG['tp_atr_mult']}×\n"
            f"Min R:R        : 1:{RISK_CONFIG['min_rr_ratio']}\n"
            f"Max daily loss : {RISK_CONFIG['max_daily_loss_pct']}%\n"
            f"Max drawdown   : {RISK_CONFIG['max_drawdown_pct']}%\n"
            f"Trailing stop  : {'ON' if RISK_CONFIG['trailing_stop'] else 'OFF'}\n"
            f"Position sizing: {RISK_CONFIG['position_sizing']}\n"
        )
        self.send(msg, chat_id=chat_id)

    # ── Pre-built message templates ───────────────────────────────────────
    def signal_generated(
        self,
        symbol:     str,
        timeframe:  str,
        direction:  str,
        confidence: float,
        xgb_dir:    str,
        lstm_dir:   str,
        smc_signal: dict = None,
        entry_price: float = None,
        atr:        float = None,
        df=None,
    ) -> None:
        if not TC["alerts"]["signal_gen"]:
            return
        smc = smc_signal or {}
        smc_bias     = smc.get("smc_bias", "neutral")
        smc_conf     = smc.get("smc_confidence", 0.0)
        smc_name     = smc.get("signal_name", "HOLD")

        dir_emoji = EMOJI["buy"] if direction == "BUY" else EMOJI["sell"]
        bias_emoji = EMOJI.get(smc_bias, EMOJI["neutral"])

        # Calculate entry/SL/TP from price & ATR
        sl_str = tp_str = entry_str = rr_str = "—"
        sl_price = tp_price = None
        if entry_price and atr and atr > 0:
            sl_mult = RISK_CONFIG["sl_atr_mult"]
            tp_mult = RISK_CONFIG["tp_atr_mult"]
            if direction == "BUY":
                sl_price = entry_price - atr * sl_mult
                tp_price = entry_price + atr * tp_mult
            else:
                sl_price = entry_price + atr * sl_mult
                tp_price = entry_price - atr * tp_mult
            rr = tp_mult / sl_mult if sl_mult else 0
            # Determine decimal places based on price magnitude
            dec = 5 if entry_price < 50 else 2
            entry_str = f"{entry_price:.{dec}f}"
            sl_str    = f"{sl_price:.{dec}f}"
            tp_str    = f"{tp_price:.{dec}f}"
            rr_str    = f"1:{rr:.1f}"

        # Confidence bar visual
        conf_bars = int(confidence * 10)
        conf_visual = "█" * conf_bars + "░" * (10 - conf_bars)

        # SMC accuracy context
        smc_accuracy_note = (
            "~65–72% when OB + sweep + BOS align"
            if smc_name != "HOLD"
            else "waiting for structure"
        )

        msg = (
            f"{dir_emoji} <b>SIGNAL: {symbol} {timeframe}</b>\n"
            f"{'─'*28}\n"
            f"Direction  : <b>{direction}</b>\n"
            f"Confidence : [{conf_visual}] {confidence:.1%}\n"
            f"XGBoost    : {xgb_dir}   |   LSTM: {lstm_dir}\n"
            f"\n"
            f"{EMOJI['money']} <b>Trade Levels</b>\n"
            f"Entry : <code>{entry_str}</code>\n"
            f"SL    : <code>{sl_str}</code>\n"
            f"TP    : <code>{tp_str}</code>\n"
            f"R:R   : <b>{rr_str}</b>\n"
            f"\n"
            f"{bias_emoji} <b>SMC Analysis</b>\n"
            f"Bias       : <b>{smc_bias.upper()}</b> ({smc_conf:.1%})\n"
            f"SMC Signal : {smc_name}\n"
            f"Accuracy   : <i>{smc_accuracy_note}</i>\n"
            f"{'─'*28}\n"
            f"{EMOJI['clock']} {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        )

        # Build inline keyboard
        keyboard = self._signal_keyboard(symbol, timeframe, direction)

        # Send chart image first if dataframe available
        if df is not None and not df.empty:
            chart_bytes = self._generate_chart(
                df, symbol, timeframe,
                entry=entry_price, sl=sl_price, tp=tp_price,
            )
            if chart_bytes:
                self.broadcast_photo(chart_bytes, caption=f"{dir_emoji} {symbol} {timeframe}")

        # Send signal message with inline keyboard and track message IDs
        msg_ids = self.broadcast(msg, reply_markup=keyboard)
        key = f"{symbol}_{timeframe}"
        self._signal_msg_ids[key] = msg_ids

    def trade_opened(self, symbol: str, direction: str, lot: float,
                     entry: float, sl: float, tp: float,
                     rr: float, confidence: float) -> None:
        if not TC["alerts"]["trade_open"]:
            return
        emoji = EMOJI["buy"] if direction == "BUY" else EMOJI["sell"]
        dec = 5 if entry < 50 else 2
        msg = (
            f"{emoji} <b>TRADE OPENED</b>\n"
            f"{'─'*28}\n"
            f"Symbol   : <code>{symbol}</code>\n"
            f"Direction: <b>{direction}</b>\n"
            f"Lot size : {lot:.2f}\n"
            f"Entry    : <code>{entry:.{dec}f}</code>\n"
            f"SL       : <code>{sl:.{dec}f}</code>\n"
            f"TP       : <code>{tp:.{dec}f}</code>\n"
            f"R:R      : 1:{rr:.1f}\n"
            f"Confidence: {confidence:.1%}\n"
            f"{'─'*28}\n"
            f"{EMOJI['clock']} {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        )
        self.broadcast(msg)

    def trade_closed(self, symbol: str, direction: str, pnl: float,
                     exit_reason: str, balance: float,
                     timeframe: str = None) -> None:
        if not TC["alerts"]["trade_close"]:
            return
        emoji = EMOJI["tp"] if exit_reason == "TP" else EMOJI["sl"]
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        result_emoji = EMOJI["fire"] if pnl >= 0 else EMOJI["bear"]

        msg = (
            f"{emoji} <b>TRADE CLOSED ({exit_reason})</b>\n"
            f"{'─'*28}\n"
            f"Symbol : <code>{symbol}</code>\n"
            f"Side   : {direction}\n"
            f"PnL    : <b>{result_emoji} {pnl_str}</b>\n"
            f"Balance: ${balance:,.2f}\n"
            f"{'─'*28}\n"
            f"{EMOJI['clock']} {datetime.utcnow().strftime('%H:%M')} UTC"
        )
        self.broadcast(msg)

        # Edit-in-place: update the original signal message
        key = f"{symbol}_{timeframe}" if timeframe else None
        if key and key in self._signal_msg_ids:
            status = "✅ CLOSED (TP)" if exit_reason == "TP" else "🛑 CLOSED (SL)"
            for cid, mid in self._signal_msg_ids[key].items():
                self.edit_message(
                    cid, mid,
                    f"{status} — PnL: <b>{pnl_str}</b>\n"
                    f"<s>Signal closed</s> at {datetime.utcnow().strftime('%H:%M')} UTC",
                )
            del self._signal_msg_ids[key]

    def kill_switch_triggered(self, reason: str, balance: float) -> None:
        if not TC["alerts"]["kill_switch"]:
            return
        msg = (
            f"{EMOJI['kill']} <b>KILL SWITCH TRIGGERED</b>\n"
            f"{'─'*28}\n"
            f"Reason : {reason}\n"
            f"Balance: ${balance:,.2f}\n"
            f"{EMOJI['clock']} {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"<i>All new trades suspended.</i>"
        )
        self.broadcast(msg)

    def daily_report(self, stats: dict) -> None:
        if not TC["alerts"]["daily_report"]:
            return
        pnl  = stats.get("daily_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        wr   = stats.get("win_rate", 0)
        wr_emoji = EMOJI["fire"] if wr >= 0.6 else (EMOJI["star"] if wr >= 0.5 else EMOJI["bear"])

        msg = (
            f"{EMOJI['report']} <b>DAILY REPORT</b> — {datetime.utcnow().date()}\n"
            f"{'─'*28}\n"
            f"Balance    : ${stats.get('balance', 0):,.2f}\n"
            f"Daily PnL  : <b>{sign}${pnl:.2f}</b>\n"
            f"Trades     : {stats.get('trades_today', 0)}\n"
            f"Win rate   : {wr_emoji} {wr:.1%}\n"
            f"Drawdown   : {stats.get('drawdown_pct', 0):.2f}%\n"
            f"Open       : {stats.get('open_trades', 0)} position(s)\n"
            f"{'─'*28}\n"
            f"Subscribers: {self.subscriber_count()}\n"
            f"{EMOJI['clock']} {datetime.utcnow().strftime('%H:%M')} UTC"
        )
        self.broadcast(msg)

    def error_alert(self, error_msg: str) -> None:
        if not TC["alerts"]["error"]:
            return
        msg = f"{EMOJI['error']} <b>BOT ERROR</b>\n<code>{error_msg[:500]}</code>"
        self.send(msg)  # errors go to owner only

    def info(self, message: str) -> None:
        self.send(f"{EMOJI['info']} {message}")  # info goes to owner only
