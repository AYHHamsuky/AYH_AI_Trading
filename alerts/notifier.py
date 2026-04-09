"""
Telegram notifier — sends trade alerts, daily reports, kill-switch warnings.
Supports multi-subscriber broadcast: users can /subscribe and /unsubscribe.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional, Set
import requests
from config import TELEGRAM_CONFIG as TC

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
}

SUBSCRIBERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "subscribers.json"
)


class TelegramNotifier:
    """
    Telegram bot notifier with multi-subscriber broadcast support.

    Users DM the bot with /subscribe to receive all signals and alerts.
    /unsubscribe removes them.  The owner chat_id always receives messages.
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

    # ── Core send ─────────────────────────────────────────────────────────
    def send(self, message: str, chat_id: str = None, parse_mode: str = "HTML") -> bool:
        if not self._enabled:
            logger.debug("Telegram disabled (no token/chat_id)")
            return False
        target = str(chat_id) if chat_id else str(self.chat_id)
        try:
            resp = requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": target, "text": message, "parse_mode": parse_mode},
                timeout=10,
            )
            if not resp.ok:
                logger.warning("Telegram send failed (chat=%s): %s", target, resp.text)
            return resp.ok
        except Exception as exc:
            logger.warning("Telegram error: %s", exc)
            return False

    def broadcast(self, message: str, parse_mode: str = "HTML") -> None:
        """Send message to owner + all subscribers."""
        recipients = {str(self.chat_id)} | self._subscribers
        for cid in recipients:
            if cid:
                self.send(message, chat_id=cid, parse_mode=parse_mode)

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

    # ── Background polling for /subscribe commands ────────────────────────
    def start_polling(self) -> None:
        """Start background thread to handle /subscribe and /unsubscribe."""
        if self._poll_thread and self._poll_thread.is_alive():
            return
        if not self._enabled:
            return
        self._poll_running = True
        self._poll_thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="tg-poll"
        )
        self._poll_thread.start()
        logger.info("Telegram polling started (subscribers: %d)", len(self._subscribers))

    def stop_polling(self) -> None:
        self._poll_running = False

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
            msg = upd.get("message") or upd.get("channel_post", {})
            if not msg:
                continue
            text    = (msg.get("text") or "").strip().lower()
            chat_id = str(msg.get("chat", {}).get("id", ""))
            if not chat_id:
                continue

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
                self.send(
                    f"{EMOJI['info']} <b>AYH Bot Status</b>\n"
                    f"Subscribers: {self.subscriber_count()}",
                    chat_id=chat_id,
                )
            elif text.startswith("/start") or text.startswith("/help"):
                self.send(
                    f"{EMOJI['signal']} <b>AYH AI Trading Bot</b>\n\n"
                    "<b>Commands:</b>\n"
                    "/subscribe — receive live signals &amp; alerts\n"
                    "/unsubscribe — stop receiving signals\n"
                    "/status — bot status\n\n"
                    "<i>Signals include SMC, XGBoost &amp; LSTM analysis.</i>",
                    chat_id=chat_id,
                )
        if updates:
            self._save_subscribers()

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
    ) -> None:
        if not TC["alerts"]["signal_gen"]:
            return
        smc = smc_signal or {}
        smc_bias     = smc.get("smc_bias", "neutral")
        smc_conf     = smc.get("smc_confidence", 0.0)
        smc_name     = smc.get("signal_name", "HOLD")

        dir_emoji = EMOJI["buy"] if direction == "BUY" else EMOJI["sell"]
        bias_emoji = EMOJI.get(smc_bias, EMOJI["neutral"])

        # Accuracy context for SMC
        smc_accuracy_note = (
            "~65–72% when OB + sweep + BOS align"
            if smc_name != "HOLD"
            else "waiting for structure"
        )

        msg = (
            f"{dir_emoji} <b>SIGNAL: {symbol} {timeframe}</b>\n"
            f"{'─'*28}\n"
            f"Direction  : <b>{direction}</b>  ({confidence:.1%} confidence)\n"
            f"XGBoost    : {xgb_dir}   |   LSTM: {lstm_dir}\n"
            f"\n"
            f"{bias_emoji} <b>SMC Analysis</b>\n"
            f"Bias       : <b>{smc_bias.upper()}</b> ({smc_conf:.1%})\n"
            f"SMC Signal : {smc_name}\n"
            f"Accuracy   : <i>{smc_accuracy_note}</i>\n"
            f"{'─'*28}\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        )
        self.broadcast(msg)

    def trade_opened(self, symbol: str, direction: str, lot: float,
                     entry: float, sl: float, tp: float,
                     rr: float, confidence: float) -> None:
        if not TC["alerts"]["trade_open"]:
            return
        emoji = EMOJI["buy"] if direction == "BUY" else EMOJI["sell"]
        msg = (
            f"{emoji} <b>TRADE OPENED</b>\n"
            f"Symbol   : <code>{symbol}</code>\n"
            f"Direction: <b>{direction}</b>\n"
            f"Lot size : {lot:.2f}\n"
            f"Entry    : {entry:.5f}\n"
            f"SL       : {sl:.5f}\n"
            f"TP       : {tp:.5f}\n"
            f"R:R      : 1:{rr:.1f}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Time     : {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        )
        self.broadcast(msg)

    def trade_closed(self, symbol: str, direction: str, pnl: float,
                     exit_reason: str, balance: float) -> None:
        if not TC["alerts"]["trade_close"]:
            return
        emoji = EMOJI["tp"] if exit_reason == "TP" else EMOJI["sl"]
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        msg = (
            f"{emoji} <b>TRADE CLOSED ({exit_reason})</b>\n"
            f"Symbol : <code>{symbol}</code>\n"
            f"Side   : {direction}\n"
            f"PnL    : <b>{pnl_str}</b>\n"
            f"Balance: ${balance:,.2f}\n"
            f"Time   : {datetime.utcnow().strftime('%H:%M')} UTC"
        )
        self.broadcast(msg)

    def kill_switch_triggered(self, reason: str, balance: float) -> None:
        if not TC["alerts"]["kill_switch"]:
            return
        msg = (
            f"{EMOJI['kill']} <b>KILL SWITCH TRIGGERED</b>\n"
            f"Reason : {reason}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Time   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"<i>All new trades suspended.</i>"
        )
        self.broadcast(msg)

    def daily_report(self, stats: dict) -> None:
        if not TC["alerts"]["daily_report"]:
            return
        pnl  = stats.get("daily_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"{EMOJI['report']} <b>DAILY REPORT</b> — {datetime.utcnow().date()}\n"
            f"Balance  : ${stats.get('balance', 0):,.2f}\n"
            f"Daily PnL: {sign}${pnl:.2f}\n"
            f"Trades   : {stats.get('trades_today', 0)}\n"
            f"Win rate : {stats.get('win_rate', 0):.1%}\n"
            f"Drawdown : {stats.get('drawdown_pct', 0):.2f}%\n"
            f"Open     : {stats.get('open_trades', 0)} position(s)\n"
            f"Subscribers: {self.subscriber_count()}"
        )
        self.broadcast(msg)

    def error_alert(self, error_msg: str) -> None:
        if not TC["alerts"]["error"]:
            return
        msg = f"{EMOJI['error']} <b>BOT ERROR</b>\n<code>{error_msg[:500]}</code>"
        self.send(msg)  # errors go to owner only

    def info(self, message: str) -> None:
        self.send(f"{EMOJI['info']} {message}")  # info goes to owner only
