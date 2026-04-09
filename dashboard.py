"""
AYH AI Trading Bot — Live Dashboard v4
Run: streamlit run dashboard.py
Powered by Deriv API (demo + live) · Real-time ticks · MT5 fallback
"""

import json
import os
import sys
import time
import logging
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Streamlit reruns this script on every interaction. If a previous run crashed
# mid-import, Python leaves sys.modules['features'] = None as a sentinel and
# subsequent imports raise KeyError instead of reimporting. Clear stale entries.
_LOCAL_PKGS = ("features", "models", "risk", "execution", "alerts", "backtest", "data", "config")
for _k in [k for k in sys.modules if k == _LOCAL_PKGS or any(k == p or k.startswith(p + ".") for p in _LOCAL_PKGS)]:
    del sys.modules[_k]

from features.engineer import FeatureEngineer
from features.smc import SMCEngine

logging.basicConfig(level=logging.WARNING)

# ── Deriv symbol groups ─────────────────────────────────────────────────────
FOREX_SYMS     = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","USDCAD","EURGBP","EURJPY"]
METAL_SYMS     = ["XAUUSD","XAGUSD"]
CRYPTO_SYMS    = ["BTCUSD","ETHUSD","LTCUSD"]
SYNTHETIC_SYMS = ["V75","V50","V25","V100","V10","BOOM1000","CRASH1000","JUMP10","JUMP25"]

ALL_SYMBOLS = FOREX_SYMS + METAL_SYMS + CRYPTO_SYMS + SYNTHETIC_SYMS

TF_OPTIONS = ["M1","M5","M15","H1","H4","D1"]

st.set_page_config(page_title="AYH AI Trader · Deriv",
                   page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
.stApp{background:#0d1117;color:#e6edf3}
section[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;margin:4px 0}
.sec{font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
     color:#8b949e;margin:10px 0 4px;display:block}
.gate-ok{color:#3fb950;font-size:.87rem;line-height:1.9}
.gate-no{color:#f85149;font-size:.87rem;line-height:1.9}
.smc-tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.75rem;
         font-weight:600;margin:2px}
.ob-bull{background:rgba(63,185,80,.15);color:#3fb950;border:1px solid #3fb950}
.ob-bear{background:rgba(248,81,73,.15);color:#f85149;border:1px solid #f85149}
.liq-tag{background:rgba(210,153,34,.15);color:#d29922;border:1px solid #d29922}
.pill-demo{background:#58a6ff;color:#000;border-radius:8px;padding:2px 10px;
           font-size:.75rem;font-weight:700}
.pill-live{background:#f85149;color:#fff;border-radius:8px;padding:2px 10px;
           font-size:.75rem;font-weight:700}
.pill-paper{background:#3fb950;color:#000;border-radius:8px;padding:2px 10px;
            font-size:.75rem;font-weight:700}
.pill-off{background:#30363d;color:#8b949e;border-radius:8px;padding:2px 10px;font-size:.75rem}
div[data-testid="stMetricValue"]{font-size:1.2rem!important}
</style>""", unsafe_allow_html=True)

# Session state defaults
for k,v in {
    "deriv_client": None, "deriv_connected": False,
    "deriv_account_info": {}, "deriv_accounts": [],
    "auto_enabled": False, "auto_mode": "paper",
    "_killed": False, "account_mode": "demo",
}.items():
    if k not in st.session_state: st.session_state[k] = v


# ── Deriv helpers ──────────────────────────────────────────────────────────
@st.cache_resource
def get_deriv_client(token: str, mode: str):
    """Create and connect a Deriv client (cached across reruns)."""
    try:
        from execution.deriv_broker import DerivClient
        client = DerivClient(token, account_type=mode)
        if client.connect():
            return client
    except Exception as exc:
        st.error(f"Deriv connection error: {exc}")
    return None


@st.cache_data(ttl=60)
def fetch_ohlcv_deriv(symbol, tf, bars, token, mode):
    """Fetch OHLCV via Deriv API (cached 60s)."""
    try:
        from execution.deriv_broker import DerivClient
        client = DerivClient(token, mode)
        if not client.connect(): return pd.DataFrame()
        df = client.get_ohlcv(symbol, tf, count=bars)
        client.disconnect()
        return df
    except Exception as exc:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_ohlcv_yf(symbol, tf, bars):
    """Fallback yfinance fetch."""
    import yfinance as yf
    YF = {"EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X",
          "AUDUSD":"AUDUSD=X","USDCHF":"USDCHF=X","XAUUSD":"GC=F",
          "BTCUSD":"BTC-USD","ETHUSD":"ETH-USD","XAGUSD":"SI=F"}
    TF = {"M1":"1m","M5":"5m","M15":"15m","H1":"1h","H4":"4h","D1":"1d"}
    yf_sym = YF.get(symbol, symbol+"=X")
    tf_str = TF.get(tf,"1h")
    tf_mins= {"M1":1,"M5":5,"M15":15,"H1":60,"H4":240,"D1":1440}
    start  = datetime.now(timezone.utc)-timedelta(minutes=tf_mins.get(tf,60)*bars)
    df = yf.download(yf_sym,start=start,interval=tf_str,progress=False,auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df[["open","high","low","close","volume"]].dropna()


@st.cache_data(ttl=60)
def analyse_df(df_json):
    df  = pd.read_json(StringIO(df_json))
    eng = FeatureEngineer()
    feat = eng.build(df, add_labels=False, add_smc=False)
    smc  = SMCEngine(swing_lookback=8)
    res  = smc.analyze(df)
    sig  = smc.get_smc_signal(df)
    return feat.to_json(), res, sig


def load_trades():
    p = "logs/paper_trades.json"
    if not os.path.exists(p): return []
    try:
        with open(p) as f: return json.load(f)
    except: return []


def load_backtest(symbol, tf):
    """Load backtest results JSON for a given symbol+timeframe."""
    p = f"logs/backtest_{symbol}_{tf}.json"
    if not os.path.exists(p): return None
    try:
        with open(p) as f: return json.load(f)
    except: return None


def load_all_backtests():
    """Scan logs/ for all backtest result files."""
    results = []
    if not os.path.exists("logs"): return results
    for fn in os.listdir("logs"):
        if fn.startswith("backtest_") and fn.endswith(".json"):
            try:
                with open(os.path.join("logs", fn)) as f:
                    results.append(json.load(f))
            except: pass
    return results


def fmt(sym, p):
    if "JPY" in sym: return f"{p:.3f}"
    if "BTC" in sym: return f"${p:,.0f}"
    if "XAU" in sym or "SI=F" in sym: return f"${p:,.2f}"
    return f"{p:.5f}"


def gate(label, ok, detail=""):
    ico=("✅" if ok else "❌")
    cls=("gate-ok" if ok else "gate-no")
    d  =f" <span style='color:#8b949e;font-size:.8rem'>{detail}</span>" if detail else ""
    return f"<div class='{cls}'>{ico} {label}{d}</div>"


def gauge(val, label, color):
    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=round(val*100,1),
        number={"suffix":"%","font":{"size":20,"color":"#e6edf3"}},
        title={"text":label,"font":{"size":10,"color":"#8b949e"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#30363d"},
               "bar":{"color":color,"thickness":0.25},
               "bgcolor":"#21262d","bordercolor":"#30363d",
               "threshold":{"line":{"color":color,"width":2},"thickness":0.75,"value":65}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=6,r=6,t=26,b=6),height=155)
    return fig


def build_chart(df, feat, smc_res, show_smc):
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
                      row_heights=[0.65,0.18,0.17],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,
        name="Price",increasing_line_color="#3fb950",increasing_fillcolor="#3fb950",
        decreasing_line_color="#f85149",decreasing_fillcolor="#f85149",line_width=1),row=1,col=1)
    for p,c,w in [(9,"#58a6ff",1),(21,"#d29922",1),(50,"#a371f7",1.5)]:
        col=f"ema_{p}"
        if col in feat.columns:
            fig.add_trace(go.Scatter(x=feat.index,y=feat[col],name=f"EMA{p}",
                line=dict(color=c,width=w),opacity=0.8),row=1,col=1)
    if "bb_upper" in feat.columns:
        fig.add_trace(go.Scatter(x=feat.index,y=feat["bb_upper"],
            line=dict(color="#30363d",width=1,dash="dot"),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=feat.index,y=feat["bb_lower"],
            line=dict(color="#30363d",width=1,dash="dot"),
            fill="tonexty",fillcolor="rgba(88,166,255,0.03)",showlegend=False),row=1,col=1)
    if show_smc and len(df)>0:
        xe=df.index[-1]+(df.index[-1]-df.index[-2])*8
        for ob in smc_res.order_blocks:
            if not ob.active or ob.index>=len(df): continue
            ts=df.index[ob.index]
            fill="#3fb950" if ob.ob_type=="bullish" else "#f85149"
            fig.add_shape(type="rect",xref="x",yref="y",x0=ts,x1=xe,y0=ob.low,y1=ob.high,
                fillcolor=f"rgba({'63,185,80' if ob.ob_type=='bullish' else '248,81,73'},.10)",
                line=dict(color=fill,width=1,dash="dot"),row=1,col=1)
            fig.add_annotation(x=ts,y=ob.high,text=f"{'🟢' if ob.ob_type=='bullish' else '🔴'} OB",
                showarrow=False,font=dict(size=9,color=fill),
                xanchor="left",yanchor="bottom",row=1,col=1)
        for sw in smc_res.liquidity_sweeps[-10:]:
            if sw.index>=len(df): continue
            col="#3fb950" if sw.sweep_type=="sellside" else "#f85149"
            mk="triangle-up" if sw.sweep_type=="sellside" else "triangle-down"
            yp=sw.sweep_low if sw.sweep_type=="sellside" else sw.sweep_high
            fig.add_trace(go.Scatter(x=[df.index[sw.index]],y=[yp],mode="markers+text",
                marker=dict(symbol=mk,size=12,color=col),
                text=["SSL↗" if sw.sweep_type=="sellside" else "BSL↘"],
                textposition="bottom center" if sw.sweep_type=="sellside" else "top center",
                textfont=dict(size=9,color=col),showlegend=False),row=1,col=1)
        for ind in smc_res.inducements[-5:]:
            if ind.index>=len(df): continue
            fig.add_hline(y=ind.level,line=dict(color="#58a6ff",width=1,dash="dashdot"),
                annotation_text="IND",annotation_font_size=8,
                annotation_font_color="#58a6ff",row=1,col=1)
        for fvg in smc_res.fair_value_gaps[-5:]:
            if fvg.index>=len(df): continue
            fig.add_shape(type="rect",xref="x",yref="y",x0=df.index[fvg.index],x1=xe,
                y0=fvg.low,y1=fvg.high,
                fillcolor="rgba(63,185,80,.07)" if fvg.fvg_type=="bullish" else "rgba(248,81,73,.07)",
                line=dict(width=0),row=1,col=1)
        for i,h in smc_res.swing_highs[-10:]:
            if i<len(df):
                fig.add_annotation(x=df.index[i],y=h,text="▼",
                    font=dict(size=8,color="#d29922"),showarrow=False,yanchor="bottom",row=1,col=1)
        for i,l in smc_res.swing_lows[-10:]:
            if i<len(df):
                fig.add_annotation(x=df.index[i],y=l,text="▲",
                    font=dict(size=8,color="#d29922"),showarrow=False,yanchor="top",row=1,col=1)
    if "rsi" in feat.columns:
        fig.add_trace(go.Scatter(x=feat.index,y=feat["rsi"],name="RSI",
            line=dict(color="#58a6ff",width=1.5)),row=2,col=1)
        for lv,c in [(70,"rgba(248,81,73,.3)"),(30,"rgba(63,185,80,.3)")]:
            fig.add_hline(y=lv,line=dict(color=c,width=1,dash="dash"),row=2,col=1)
    if "macd_hist" in feat.columns:
        hist=feat["macd_hist"]
        fig.add_trace(go.Bar(x=feat.index,y=hist,name="MACD Hist",
            marker_color=["#3fb950" if v>=0 else "#f85149" for v in hist],opacity=0.7),row=3,col=1)
        if "macd" in feat.columns:
            fig.add_trace(go.Scatter(x=feat.index,y=feat["macd"],name="MACD",
                line=dict(color="#58a6ff",width=1.2)),row=3,col=1)
        if "macd_signal" in feat.columns:
            fig.add_trace(go.Scatter(x=feat.index,y=feat["macd_signal"],name="Signal",
                line=dict(color="#d29922",width=1.2)),row=3,col=1)
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
        xaxis_rangeslider_visible=False,height=660,margin=dict(l=6,r=6,t=26,b=6),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
                    font=dict(size=10),bgcolor="rgba(0,0,0,0)"),
        font=dict(family="monospace",size=11,color="#8b949e"))
    fig.update_yaxes(gridcolor="#21262d",zerolinecolor="#21262d")
    fig.update_xaxes(gridcolor="#21262d",showspikes=True,spikecolor="#30363d")
    return fig


# ══════════════════════════════════════════════════════════════════════════
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚡ AYH AI Trader")
        st.markdown("---")

        # ── Deriv account connection ──────────────────────────────────────
        st.markdown('<span class="sec">🔑 Deriv Account</span>', unsafe_allow_html=True)
        account_mode = st.radio("Account", ["📘 Demo", "💰 Live"], index=0, horizontal=True,
                                help="Demo = practice account (free), Live = real money")
        mode_str = "demo" if "Demo" in account_mode else "live"

        token_key = f"DERIV_API_TOKEN_{mode_str.upper()}"
        token_env = os.getenv(token_key, "")
        manual_token = st.text_input(
            "API Token (or set in .env)",
            type="password",
            value="",
            placeholder=f"leave blank — reads {token_key} from .env",
        )
        use_token = manual_token.strip() or token_env

        if st.button("🔌 Connect", width="stretch"):
            if not use_token:
                st.error(f"Set {token_key} in .env or enter token above")
            else:
                with st.spinner("Connecting to Deriv..."):
                    try:
                        from execution.deriv_broker import DerivClient
                        client = DerivClient(use_token, mode_str)
                        if client.connect():
                            st.session_state.deriv_client     = client
                            st.session_state.deriv_connected  = True
                            st.session_state.account_mode     = mode_str
                            info = client.account_info()
                            accs = client.get_accounts()
                            st.session_state.deriv_account_info = info
                            st.session_state.deriv_accounts     = accs
                            st.success(
                                f"✅ Connected!\n"
                                f"Balance: {info.get('currency','')} "
                                f"{info.get('balance',0):,.2f}\n"
                                f"Virtual: {info.get('is_virtual',True)}"
                            )
                        else:
                            st.error("Connection failed — check token")
                    except ImportError:
                        st.error("Run: pip install websocket-client")

        # Show account info if connected
        if st.session_state.deriv_connected:
            info = st.session_state.deriv_account_info
            mode_label = "DEMO" if info.get("is_virtual") else "LIVE"
            st.markdown(
                f'<div style="background:rgba(88,166,255,.1);border:1px solid #58a6ff;'
                f'border-radius:8px;padding:8px 12px;margin:6px 0;font-size:.82rem">'
                f'<b style="color:#58a6ff">{mode_label}</b> · '
                f'{info.get("currency","")} {info.get("balance",0):,.2f}<br>'
                f'<span style="color:#8b949e">{info.get("login_id","")}</span>'
                f'</div>', unsafe_allow_html=True
            )

            # Account switcher
            if len(st.session_state.deriv_accounts) > 1:
                st.markdown('<span class="sec">Switch Account</span>', unsafe_allow_html=True)
                acc_options = {
                    f"{a['login_id']} ({a['account_type'].upper()} {a['currency']})": a["login_id"]
                    for a in st.session_state.deriv_accounts
                }
                selected_acc = st.selectbox("Account", list(acc_options.keys()), label_visibility="collapsed")
                if st.button("Switch", width="stretch"):
                    client = st.session_state.deriv_client
                    if client and client.switch_account(acc_options[selected_acc]):
                        new_info = client.account_info()
                        st.session_state.deriv_account_info = new_info
                        st.rerun()

        st.markdown("---")

        # ── Symbol & timeframe ────────────────────────────────────────────
        sym_group = st.selectbox("Category", ["Forex","Metals","Crypto","Synthetics (Deriv)"])
        sym_map   = {"Forex":FOREX_SYMS,"Metals":METAL_SYMS,
                     "Crypto":CRYPTO_SYMS,"Synthetics (Deriv)":SYNTHETIC_SYMS}
        symbol    = st.selectbox("Symbol",    sym_map[sym_group])
        timeframe = st.selectbox("Timeframe", TF_OPTIONS, index=3)
        bars      = st.slider("Bars", 100, 1000, 400, 50)
        show_smc  = st.toggle("SMC overlay", value=True)

        st.markdown("---")
        st.markdown('<span class="sec">🤖 Auto-Execution</span>', unsafe_allow_html=True)

        exec_mode = st.radio("Execution",["📄 Paper","🔴 Live Orders"],
                             index=0, horizontal=True)
        exec_str  = "paper" if "Paper" in exec_mode else "live"
        if exec_str == "live":
            if info.get("is_virtual", True) if st.session_state.deriv_connected else True:
                st.info("Using DEMO account — safe to enable live orders")
            else:
                st.warning("⚠️ LIVE account — real money at risk!")

        auto_on = st.toggle("Enable AutoTrader", value=st.session_state.auto_enabled)
        if auto_on != st.session_state.auto_enabled:
            st.session_state.auto_enabled = auto_on
            st.session_state.auto_mode    = exec_str

        st.markdown('<span class="sec">Gate Thresholds</span>', unsafe_allow_html=True)
        min_ens  = st.slider("Min ensemble conf", 0.50, 0.90, 0.68, 0.01)
        min_smc  = st.slider("Min SMC conf",      0.40, 0.90, 0.60, 0.01)

        st.markdown("---")
        auto_ref = st.toggle("Auto refresh",value=False)
        ref_s    = st.select_slider("Interval",[30,60,120,300],value=60)

        if st.button("🚨 Kill Switch", width="stretch"):
            st.session_state.auto_enabled = False
            st.session_state._killed      = True
            st.warning("Kill switch triggered")

        st.caption(f"🕐 {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
        st.caption("Deriv API · Mac-native · No MT5 bridge needed")

    # ── Header ───────────────────────────────────────────────────────────
    acc_pill = (
        f'<span class="pill-demo">📘 DEMO</span>'
        if st.session_state.account_mode == "demo"
        else f'<span class="pill-live">💰 LIVE</span>'
    ) if st.session_state.deriv_connected else '<span class="pill-off">⚪ NOT CONNECTED</span>'

    auto_pill = (
        f'<span class="pill-paper">● PAPER</span>'
        if exec_str == "paper" and st.session_state.auto_enabled
        else f'<span class="pill-live">● LIVE ORDERS</span>'
        if exec_str == "live" and st.session_state.auto_enabled
        else '<span class="pill-off">● AUTO OFF</span>'
    )

    hc1, hc2 = st.columns([5,1])
    with hc1:
        st.markdown(
            f"## 📊 {symbol} · {timeframe} &nbsp; {acc_pill} &nbsp; {auto_pill}",
            unsafe_allow_html=True
        )
    with hc2:
        if st.button("🔄 Refresh", width="stretch"):
            st.cache_data.clear(); st.rerun()

    # ── Fetch data ────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {symbol} {timeframe} data..."):
        df_raw = pd.DataFrame()
        # Try Deriv first (required for synthetics)
        if use_token and symbol in SYNTHETIC_SYMS:
            df_raw = fetch_ohlcv_deriv(symbol, timeframe, bars, use_token, mode_str)
        elif use_token:
            df_raw = fetch_ohlcv_deriv(symbol, timeframe, bars, use_token, mode_str)
        if df_raw.empty:
            df_raw = fetch_ohlcv_yf(symbol, timeframe, bars)

    data_src = "Deriv" if (use_token and not df_raw.empty and
                            len(df_raw)>10) else "yfinance"

    if df_raw.empty:
        st.error(f"❌ No data for {symbol}. "
                 f"{'Connect Deriv above for synthetic indices.' if symbol in SYNTHETIC_SYMS else 'Check connection.'}")
        st.stop()

    with st.spinner("Computing indicators + SMC..."):
        feat_json, smc_res, smc_sig = analyse_df(df_raw.to_json())
        df_feat = pd.read_json(StringIO(feat_json))

    # ── Metrics strip ─────────────────────────────────────────────────────
    cur   = float(df_raw["close"].iloc[-1])
    prv   = float(df_raw["close"].iloc[-2])
    chg   = (cur-prv)/prv*100
    atr   = float(df_feat["atr"].iloc[-1]) if "atr" in df_feat.columns else cur*0.001
    rsi   = float(df_feat["rsi"].iloc[-1]) if "rsi" in df_feat.columns else 50.0
    h_now = datetime.now(timezone.utc).hour

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Price",     fmt(symbol,cur), f"{chg:+.3f}%",
              delta_color="normal" if chg>=0 else "inverse")
    m2.metric("ATR",       fmt(symbol,atr))
    m3.metric("RSI",       f"{rsi:.1f}")
    m4.metric("SMC Bias",  smc_res.bias.upper(), f"{smc_res.bias_confidence:.0%}")
    m5.metric("SMC Signal",smc_sig["signal_name"], f"{smc_sig['confidence']:.0%}")
    m6.metric("Data Source", data_src, f"{len(df_raw)} bars")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────
    t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 = st.tabs([
        "📈 Chart","🧠 SMC","🔔 Signals",
        "🤖 AutoTrader","💼 Account","📋 Trade Log","⚖️ Risk",
        "📊 Backtest","🚀 Live Performance","📲 Telegram"
    ])

    # ══ CHART ══════════════════════════════════════════════════════════════
    with t1:
        st.plotly_chart(build_chart(df_raw,df_feat,smc_res,show_smc),
                        width="stretch",config={"displayModeBar":False})
        if show_smc:
            lc=st.columns(5)
            lc[0].markdown('<span class="smc-tag ob-bull">🟩 Bullish OB</span>',unsafe_allow_html=True)
            lc[1].markdown('<span class="smc-tag ob-bear">🟥 Bearish OB</span>',unsafe_allow_html=True)
            lc[2].markdown('<span class="smc-tag liq-tag">🔶 Liq. Sweep</span>',unsafe_allow_html=True)
            lc[3].markdown('<span class="smc-tag" style="background:rgba(88,166,255,.15);color:#58a6ff;border:1px solid #58a6ff">🔷 Inducement</span>',unsafe_allow_html=True)
            lc[4].markdown('<span class="smc-tag" style="background:rgba(130,80,255,.15);color:#a371f7;border:1px solid #a371f7">🟪 FVG</span>',unsafe_allow_html=True)

    # ══ SMC ════════════════════════════════════════════════════════════════
    with t2:
        bc={"bullish":"#3fb950","bearish":"#f85149","neutral":"#d29922"}.get(smc_res.bias,"#8b949e")
        bi={"bullish":"🐂","bearish":"🐻","neutral":"🦀"}.get(smc_res.bias,"•")
        st.markdown(f"""
        <div style="background:rgba(0,0,0,.3);border:1px solid {bc};border-radius:10px;
             padding:12px 18px;margin-bottom:12px;display:flex;align-items:center;gap:12px">
          <span style="font-size:1.8rem">{bi}</span>
          <div><div style="color:{bc};font-size:1.05rem;font-weight:700">
            {smc_res.bias.upper()} BIAS · {smc_res.bias_confidence:.0%}</div>
            <div style="color:#8b949e;font-size:.83rem">
              SMC Signal: <b style="color:{bc}">{smc_sig["signal_name"]}</b>
              ({smc_sig["confidence"]:.0%})</div></div></div>""",unsafe_allow_html=True)

        s1,s2=st.columns(2)
        with s1:
            st.markdown('<span class="sec">Active Order Blocks</span>',unsafe_allow_html=True)
            aobs=[o for o in smc_res.order_blocks if o.active]
            if aobs:
                st.dataframe(pd.DataFrame([{"Type":"🟢" if o.ob_type=="bullish" else "🔴",
                    "High":fmt(symbol,o.high),"Low":fmt(symbol,o.low),"Impulse":f"{o.impulse_size:.2%}"}
                    for o in sorted(aobs,key=lambda x:x.index,reverse=True)[:8]]),
                    width="stretch",hide_index=True)
            else: st.info("No active OBs")
        with s2:
            st.markdown('<span class="sec">Liquidity Sweeps (confirmed)</span>',unsafe_allow_html=True)
            csw=[s for s in smc_res.liquidity_sweeps if s.confirmed]
            if csw:
                st.dataframe(pd.DataFrame([{"Type":"↗ Sell-side" if s.sweep_type=="sellside" else "↘ Buy-side",
                    "Level":fmt(symbol,s.level),"Reversal":f"{s.reversal_size:.2%}"}
                    for s in sorted(csw,key=lambda x:x.index,reverse=True)[:8]]),
                    width="stretch",hide_index=True)
            else: st.info("No confirmed sweeps")
        s3,s4=st.columns(2)
        with s3:
            st.markdown('<span class="sec">Swept Inducements</span>',unsafe_allow_html=True)
            si=[i for i in smc_res.inducements if i.swept]
            if si:
                st.dataframe(pd.DataFrame([{"Type":"↑ Bull" if i.ind_type=="bullish" else "↓ Bear",
                    "Level":fmt(symbol,i.level)}
                    for i in sorted(si,key=lambda x:x.index,reverse=True)[:8]]),
                    width="stretch",hide_index=True)
            else: st.info("No swept inducements")
        with s4:
            st.markdown('<span class="sec">Fair Value Gaps</span>',unsafe_allow_html=True)
            if smc_res.fair_value_gaps:
                st.dataframe(pd.DataFrame([{"Type":"🟢" if f.fvg_type=="bullish" else "🔴",
                    "High":fmt(symbol,f.high),"Low":fmt(symbol,f.low)}
                    for f in sorted(smc_res.fair_value_gaps,key=lambda x:x.index,reverse=True)[:8]]),
                    width="stretch",hide_index=True)
            else: st.info("No FVGs")

    # ══ SIGNALS ════════════════════════════════════════════════════════════
    with t3:
        smc_d=smc_sig["signal_name"]; smc_c=smc_sig["confidence"]
        smc_cl={"BUY":"#3fb950","SELL":"#f85149","HOLD":"#d29922"}.get(smc_d,"#58a6ff")
        rsi_s ="BUY" if rsi<35 else "SELL" if rsi>65 else "HOLD"
        rsi_c =abs(rsi-50)/50
        rsi_cl={"BUY":"#3fb950","SELL":"#f85149","HOLD":"#d29922"}[rsi_s]
        mh    =float(df_feat["macd_hist"].iloc[-1]) if "macd_hist" in df_feat.columns else 0
        mac_s ="BUY" if mh>0 else "SELL"
        mac_cl="#3fb950" if mh>0 else "#f85149"
        mac_c =min(abs(mh)/(abs(df_feat["macd_hist"]).mean()+1e-9),1.0) if "macd_hist" in df_feat.columns else 0.5
        g1,g2,g3=st.columns(3)
        g1.plotly_chart(gauge(smc_c,f"SMC ({smc_d})",smc_cl),width="stretch")
        g2.plotly_chart(gauge(rsi_c,f"RSI {rsi:.0f} ({rsi_s})",rsi_cl),width="stretch")
        g3.plotly_chart(gauge(mac_c,f"MACD ({mac_s})",mac_cl),width="stretch")
        votes=Counter([smc_d,rsi_s,mac_s]); cons=votes.most_common(1)[0][0]
        cons_cl={"BUY":"#3fb950","SELL":"#f85149","HOLD":"#d29922"}.get(cons,"#58a6ff")
        cons_ico={"BUY":"📈","SELL":"📉","HOLD":"⏸️"}.get(cons,"•")
        st.markdown(f"""
        <div style="text-align:center;background:rgba(0,0,0,.3);border:2px solid {cons_cl};
             border-radius:12px;padding:16px;margin:10px 0">
          <div style="font-size:1.8rem">{cons_ico}</div>
          <div style="color:{cons_cl};font-size:1.6rem;font-weight:700">{cons}</div>
          <div style="color:#8b949e;font-size:.87rem">
            {votes[cons]}/3 indicators · SMC:{smc_d} RSI:{rsi_s} MACD:{mac_s}</div>
        </div>""",unsafe_allow_html=True)
        if cons!="HOLD":
            mult=1 if cons=="BUY" else -1
            sl_=cur-mult*atr*1.5; tp1=cur+mult*atr*2.0; tp2=cur+mult*atr*3.5
            lv1,lv2,lv3,lv4=st.columns(4)
            lv1.metric("Entry",fmt(symbol,cur))
            lv2.metric("Stop Loss",fmt(symbol,sl_),"ATR×1.5",delta_color="inverse")
            lv3.metric("TP1",fmt(symbol,tp1),"RR 1:1.3")
            lv4.metric("TP2",fmt(symbol,tp2),"RR 1:2.3")

    # ══ AUTOTRADER ═════════════════════════════════════════════════════════
    with t4:
        ac=("#f85149" if exec_str=="live" and st.session_state.auto_enabled
            else "#3fb950" if st.session_state.auto_enabled else "#30363d")
        al=(f"{'🔴 LIVE ORDERS' if exec_str=='live' else '📄 PAPER'} — AUTOTRADER {'ACTIVE' if st.session_state.auto_enabled else 'DISABLED'}")
        st.markdown(f"""
        <div style="background:rgba(0,0,0,.3);border:2px solid {ac};border-radius:12px;
             padding:12px;margin-bottom:14px;text-align:center">
          <div style="color:{ac};font-size:1.1rem;font-weight:700">{al}</div>
          <div style="color:#8b949e;font-size:.8rem;margin-top:4px">
            Toggle "Enable AutoTrader" in sidebar · {"Deriv DEMO — safe" if st.session_state.account_mode=="demo" else "Deriv LIVE — real money"}</div>
        </div>""",unsafe_allow_html=True)

        st.markdown('<span class="sec">Live Quality Gate Check</span>',unsafe_allow_html=True)
        votes2=Counter([smc_d,rsi_s,mac_s]); cons2=votes2.most_common(1)[0][0]
        ens_c=max(smc_c,rsi_c,mac_c)
        sess_ok=(7<=h_now<16) or (12<=h_now<21)
        g_ens =ens_c>=min_ens; g_smc=smc_c>=min_smc
        g_dir =smc_d==cons2 if cons2!="HOLD" else True
        g_hold=cons2!="HOLD"; g_sess=sess_ok
        g_kill=not st.session_state._killed
        g_conn=st.session_state.deriv_connected
        all_ok=all([g_ens,g_smc,g_dir,g_hold,g_sess,g_kill,g_conn])
        gates="".join([
            gate("Deriv connected",      g_conn, f"{'✓' if g_conn else 'Connect in sidebar'}"),
            gate("Ensemble confidence",  g_ens,  f"{ens_c:.0%} (≥{min_ens:.0%})"),
            gate("SMC confidence",       g_smc,  f"{smc_c:.0%} (≥{min_smc:.0%})"),
            gate("SMC agrees",           g_dir,  f"SMC={smc_d} · Consensus={cons2}"),
            gate("Signal not HOLD",      g_hold, cons2),
            gate("Active session",       g_sess, f"UTC {h_now}:xx"),
            gate("Kill switch clear",    g_kill, ""),
        ])
        st.markdown(f'<div class="card">{gates}</div>',unsafe_allow_html=True)
        if all_ok and st.session_state.auto_enabled:
            mult2=1 if cons2=="BUY" else -1
            dc={"BUY":"#3fb950","SELL":"#f85149"}.get(cons2,"#d29922")
            st.markdown(f"""
            <div style="background:rgba(63,185,80,.07);border:1px solid #3fb950;
                 border-radius:10px;padding:12px;margin:10px 0;text-align:center">
              <div style="color:#3fb950;font-weight:700">✅ ALL GATES PASSED</div>
              <div style="color:#8b949e;font-size:.85rem">
                <b style="color:{dc}">{cons2}</b> {symbol}
                @ {fmt(symbol,cur)} · mode={exec_str.upper()}</div>
            </div>""",unsafe_allow_html=True)

    # ══ ACCOUNT ════════════════════════════════════════════════════════════
    with t5:
        st.markdown("### 💼 Deriv Account")
        if st.session_state.deriv_connected:
            info=st.session_state.deriv_account_info
            accs=st.session_state.deriv_accounts
            a1,a2,a3,a4=st.columns(4)
            a1.metric("Balance",  f"{info.get('currency','')} {info.get('balance',0):,.2f}")
            a2.metric("Account",  info.get("login_id",""))
            a3.metric("Type",     "DEMO" if info.get("is_virtual") else "LIVE")
            a4.metric("Email",    info.get("email","")[:20])
            if accs:
                st.markdown("---")
                st.markdown('<span class="sec">All Linked Accounts</span>',unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{
                    "Login":   a["login_id"],
                    "Type":    a["account_type"].upper(),
                    "Currency":a["currency"],
                    "Balance": f"{a.get('balance',0):,.2f}",
                } for a in accs]),width="stretch",hide_index=True)
            # Open contracts
            st.markdown("---")
            st.markdown('<span class="sec">Open Contracts</span>',unsafe_allow_html=True)
            if st.button("🔄 Load Open Positions"):
                client=st.session_state.deriv_client
                if client:
                    contracts=client.get_open_contracts()
                    if contracts:
                        st.dataframe(pd.DataFrame(contracts),width="stretch",hide_index=True)
                    else:
                        st.info("No open contracts")
        else:
            st.info("Connect your Deriv account in the sidebar to see account details.")
            st.markdown("""
**How to get your Deriv API token (2 minutes):**

1. Go to **[app.deriv.com](https://app.deriv.com)** and log in
2. Click your profile → **Security & Safety → API Token**
3. Click **Create** — name it `AI Bot`, select all 4 scopes (Read, Trade, Payments, Admin)
4. Copy the token
5. Add to your `.env` file:
```
DERIV_API_TOKEN_DEMO=your_demo_token_here
DERIV_API_TOKEN_LIVE=your_live_token_here
```
6. Or paste directly in the sidebar above

**Why Deriv works perfectly for you:**
- ✅ Accepts Nigerian accounts
- ✅ Demo + Live account on same platform
- ✅ Free real-time data API (no Bloomberg/Reuters fees)
- ✅ Synthetic indices trade 24/7 (no weekend gaps, no news events)
- ✅ Works on Mac without any MT5 bridge
            """)

    # ══ TRADE LOG ══════════════════════════════════════════════════════════
    with t6:
        st.markdown("### 📋 Trade Log")
        # Local paper trades
        trades=load_trades()
        if trades:
            df_t=pd.DataFrame(trades)
            pnl_c=df_t["pnl"].sum() if "pnl" in df_t.columns else 0
            wins_c=(df_t["pnl"]>0).sum() if "pnl" in df_t.columns else 0
            tl1,tl2,tl3,tl4=st.columns(4)
            tl1.metric("Total",len(df_t))
            tl2.metric("Win Rate",f"{wins_c/(len(df_t)+1e-9):.0%}")
            tl3.metric("Total PnL",f"${pnl_c:+.2f}",delta_color="normal" if pnl_c>=0 else "inverse")
            tl4.metric("Mode",df_t["mode"].iloc[-1].upper() if "mode" in df_t.columns else "–")
            show_cols=[c for c in ["timestamp","symbol","direction","mode","lot_size",
                "entry_price","sl","tp","confidence","status","pnl","smc_bias"] if c in df_t.columns]
            st.dataframe(df_t[show_cols].sort_values("timestamp",ascending=False).head(50),
                         width="stretch",hide_index=True)
            st.download_button("⬇️ Export CSV",df_t.to_csv(index=False).encode(),"trades.csv","text/csv")
        # Deriv live trade history
        if st.session_state.deriv_connected:
            st.markdown("---")
            st.markdown('<span class="sec">Deriv Contract History</span>',unsafe_allow_html=True)
            if st.button("🔄 Load Deriv History"):
                client=st.session_state.deriv_client
                if client:
                    history=client.get_trade_history(50)
                    if history:
                        st.dataframe(pd.DataFrame(history),width="stretch",hide_index=True)
                    else:
                        st.info("No closed contracts found")
        if not trades and not st.session_state.deriv_connected:
            st.info("No trades yet. Connect Deriv and enable AutoTrader to start logging.")

    # ══ RISK ═══════════════════════════════════════════════════════════════
    with t7:
        st.markdown("### ⚖️ Risk Dashboard")
        r1,r2,r3,r4=st.columns(4)
        r1.metric("ATR SL",fmt(symbol,atr*1.5),"ATR×1.5")
        r2.metric("ATR TP",fmt(symbol,atr*3.0),"RR 1:2")
        r3.metric("Volatility",f"{atr/cur*100:.3f}%","of price")
        r4.metric("Session","London+NY" if 7<=h_now<16 and 12<=h_now<21
                  else "London" if 7<=h_now<16 else "New York" if 12<=h_now<21
                  else "Off-hours",f"UTC {h_now}:00")
        st.markdown("---")
        st.markdown('<span class="sec">Position Sizing Calculator</span>',unsafe_allow_html=True)
        b2=st.number_input("Balance ($)",10,100_000,
            max(10,int(st.session_state.deriv_account_info.get("balance",500))),10,key="r_bal")
        p2=st.slider("Risk %",0.1,10.0,2.0,0.1,key="r_pct")
        sl2=st.slider("SL ATR mult",0.5,3.0,1.5,0.1)
        sl_sz=atr*sl2; ra=b2*p2/100
        lot2=max(1.0,min(round(ra/sl_sz,2),500)) if sl_sz>0 else 1.0
        rc1,rc2,rc3,rc4=st.columns(4)
        rc1.metric("Risk Amount",f"${ra:.2f}")
        rc2.metric("SL Distance",fmt(symbol,sl_sz))
        rc3.metric("Stake / Lot",f"${lot2:.2f}")
        rc4.metric("TP Target",fmt(symbol,sl_sz*2))
        st.caption("For Deriv, 'Stake' = amount invested per trade. "
                   "Use Deriv multipliers to control your effective position size.")

    # ══ BACKTEST PERFORMANCE ═══════════════════════════════════════════════
    with t8:
        st.markdown("### 📊 Backtest Performance")
        
        # ── Run backtest section ───────────────────────────────────────────
        st.markdown('<span class="sec">🔄 Run New Backtest</span>', unsafe_allow_html=True)
        
        bt_cols = st.columns([3, 1, 1, 1])
        with bt_cols[0]:
            bt_symbol = st.selectbox(
                "Symbol", ALL_SYMBOLS, 
                index=ALL_SYMBOLS.index(symbol) if symbol in ALL_SYMBOLS else 0,
                key="bt_symbol", label_visibility="collapsed"
            )
        with bt_cols[1]:
            bt_tf = st.selectbox(
                "TF", TF_OPTIONS,
                index=TF_OPTIONS.index(timeframe) if timeframe in TF_OPTIONS else 3,
                key="bt_tf", label_visibility="collapsed"
            )
        with bt_cols[2]:
            bt_bars = st.number_input(
                "Bars", 500, 5000, 2000, 100, key="bt_bars", label_visibility="collapsed"
            )
        with bt_cols[3]:
            run_bt = st.button("▶️ Run", key="btn_backtest", help="Run backtest on demo account data", width="stretch")
        
        if run_bt:
            with st.spinner(f"⏳ Backtesting {bt_symbol} {bt_tf} ({int(bt_bars)} bars)..."):
                try:
                    from data.fetcher import DataFetcher
                    from backtest.engine import BacktestEngine
                    
                    # Fetch data
                    fetcher = DataFetcher(
                        deriv_token=use_token or token_env,
                        deriv_mode=mode_str,
                    )
                    df_raw = fetcher.fetch(bt_symbol, bt_tf, bars=int(bt_bars))
                    
                    if df_raw.empty:
                        st.error(f"❌ Could not fetch data for {bt_symbol}")
                    else:
                        # Run backtest
                        engine = BacktestEngine(bt_symbol, bt_tf)
                        results = engine.run(df_raw, n_windows=3)
                        
                        # Save results
                        from datetime import datetime as dt
                        import json
                        os.makedirs("logs", exist_ok=True)
                        results_file = f"logs/backtest_{bt_symbol}_{bt_tf}.json"
                        with open(results_file, "w") as f:
                            json.dump({
                                "symbol": bt_symbol,
                                "timeframe": bt_tf,
                                "run_time": dt.now().isoformat(),
                                "initial_balance": engine._balance,
                                "final_balance": engine._balance,
                                "metrics": results,
                                "trades": [asdict(t) for t in engine._trades],
                                "equity_curve": engine._equity_curve,
                            }, f, indent=2, default=str)
                        
                        st.success(f"✅ Backtest complete! Saved to {results_file}")
                        st.rerun()
                        
                except Exception as exc:
                    st.error(f"❌ Backtest failed: {str(exc)}")
        
        st.markdown("---")
        
        all_bt = load_all_backtests()
        bt_data = load_backtest(symbol, timeframe)

        if not all_bt:
            st.info(
                "No backtest results found. Run a backtest first:\n\n"
                "```bash\npython main.py --mode backtest --symbols V75 --tf H1\n```"
            )
        else:
            # Selector for available backtests
            bt_labels = [f"{b['symbol']} {b['timeframe']} — {b.get('run_time','')[:16]}" for b in all_bt]
            sel_idx = 0
            for i, b in enumerate(all_bt):
                if b["symbol"] == symbol and b["timeframe"] == timeframe:
                    sel_idx = i
                    break
            selected_bt = st.selectbox("Select backtest run", bt_labels, index=sel_idx)
            bt_data = all_bt[bt_labels.index(selected_bt)]

            if bt_data:
                met = bt_data.get("metrics", {})
                trades_bt = bt_data.get("trades", [])
                eq_curve = bt_data.get("equity_curve", [])

                # ── Key metrics cards ──────────────────────────────────
                bm1, bm2, bm3, bm4, bm5, bm6 = st.columns(6)
                total_pnl = met.get("total_pnl", 0)
                pnl_color = "normal" if total_pnl >= 0 else "inverse"
                bm1.metric("Total Trades", int(met.get("trades", 0)))
                bm2.metric("Win Rate", f"{met.get('win_rate', 0):.1%}")
                bm3.metric("Profit Factor", f"{met.get('profit_factor', 0):.2f}")
                bm4.metric("Total PnL", f"${total_pnl:,.2f}", delta_color=pnl_color)
                bm5.metric("Final Balance", f"${bt_data.get('final_balance', 10000):,.2f}")
                sharpe = met.get("sharpe", None)
                bm6.metric("Sharpe Ratio", f"{sharpe:.3f}" if sharpe else "N/A")

                st.markdown("---")

                # ── Equity curve ───────────────────────────────────────
                if eq_curve and len(eq_curve) > 1:
                    st.markdown('<span class="sec">Equity Curve</span>', unsafe_allow_html=True)
                    init_bal = bt_data.get("initial_balance", 10000)
                    eq_full = [init_bal] + eq_curve
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=list(range(len(eq_full))),
                        y=eq_full,
                        mode="lines",
                        name="Equity",
                        fill="tozeroy",
                        line=dict(color="#3fb950" if eq_full[-1] >= init_bal else "#f85149", width=2),
                        fillcolor="rgba(63,185,80,0.1)" if eq_full[-1] >= init_bal else "rgba(248,81,73,0.1)",
                    ))
                    fig_eq.add_hline(
                        y=init_bal,
                        line=dict(color="#58a6ff", width=1, dash="dash"),
                        annotation_text=f"Initial ${init_bal:,.0f}",
                        annotation_font_color="#58a6ff",
                    )
                    fig_eq.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        height=300,
                        margin=dict(l=6, r=6, t=10, b=30),
                        xaxis_title="Trade #",
                        yaxis_title="Balance ($)",
                        yaxis=dict(gridcolor="#21262d"),
                        xaxis=dict(gridcolor="#21262d"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_eq, width="stretch")

                # ── Win/Loss distribution ──────────────────────────────
                if trades_bt:
                    st.markdown('<span class="sec">Trade PnL Distribution</span>', unsafe_allow_html=True)
                    pnls_bt = [t["pnl"] for t in trades_bt]
                    colors_bt = ["#3fb950" if p > 0 else "#f85149" for p in pnls_bt]
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Bar(
                        x=list(range(1, len(pnls_bt) + 1)),
                        y=pnls_bt,
                        marker_color=colors_bt,
                        name="PnL",
                    ))
                    fig_pnl.add_hline(y=0, line=dict(color="#30363d", width=1))
                    fig_pnl.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        height=250,
                        margin=dict(l=6, r=6, t=10, b=30),
                        xaxis_title="Trade #",
                        yaxis_title="PnL ($)",
                        yaxis=dict(gridcolor="#21262d"),
                        xaxis=dict(gridcolor="#21262d"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_pnl, width="stretch")

                    # ── Detailed stats ─────────────────────────────────
                    bs1, bs2 = st.columns(2)
                    with bs1:
                        st.markdown('<span class="sec">Win / Loss Breakdown</span>', unsafe_allow_html=True)
                        wins_bt = [t for t in trades_bt if t["pnl"] > 0]
                        losses_bt = [t for t in trades_bt if t["pnl"] <= 0]
                        st.markdown(f"""
                        <div class="card">
                            <div style="color:#3fb950">✅ Wins: <b>{len(wins_bt)}</b> — Avg: <b>${np.mean([t['pnl'] for t in wins_bt]):,.2f}</b></div>
                            <div style="color:#f85149">❌ Losses: <b>{len(losses_bt)}</b> — Avg: <b>${np.mean([t['pnl'] for t in losses_bt]):,.2f}</b></div>
                            <div style="color:#8b949e;margin-top:8px">Max win: <b>${met.get('max_win',0):,.2f}</b> · Max loss: <b>${met.get('max_loss',0):,.2f}</b></div>
                            <div style="color:#8b949e">Avg win: <b>${met.get('avg_win',0):,.2f}</b> · Avg loss: <b>${met.get('avg_loss',0):,.2f}</b></div>
                        </div>""", unsafe_allow_html=True)

                    with bs2:
                        st.markdown('<span class="sec">Exit Reasons</span>', unsafe_allow_html=True)
                        reasons = Counter([t.get("exit_reason", "?") for t in trades_bt])
                        reason_labels = list(reasons.keys())
                        reason_vals = list(reasons.values())
                        fig_pie = go.Figure(go.Pie(
                            labels=reason_labels, values=reason_vals,
                            marker=dict(colors=["#3fb950", "#f85149", "#58a6ff", "#d29922"]),
                            hole=0.5,
                            textinfo="label+percent",
                            textfont=dict(color="#e6edf3"),
                        ))
                        fig_pie.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            height=200, margin=dict(l=6, r=6, t=6, b=6),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_pie, width="stretch")

                    # ── Trade list ─────────────────────────────────────
                    st.markdown("---")
                    st.markdown('<span class="sec">All Backtest Trades</span>', unsafe_allow_html=True)
                    df_bt = pd.DataFrame(trades_bt)
                    show_bt_cols = [c for c in ["entry_time", "direction", "entry_price",
                        "exit_price", "stop_loss", "take_profit", "lot_size", "pnl",
                        "exit_reason", "confidence"] if c in df_bt.columns]
                    st.dataframe(
                        df_bt[show_bt_cols].sort_values("entry_time", ascending=False),
                        width="stretch", hide_index=True
                    )
                    st.download_button(
                        "⬇️ Export Backtest CSV",
                        df_bt.to_csv(index=False).encode(),
                        f"backtest_{bt_data['symbol']}_{bt_data['timeframe']}.csv",
                        "text/csv"
                    )

    # ══ LIVE PERFORMANCE ═══════════════════════════════════════════════════
    with t9:
        st.markdown("### 🚀 Live Performance")

        if not st.session_state.deriv_connected:
            st.info("Connect your Deriv account in the sidebar to track live performance.")
        else:
            info = st.session_state.deriv_account_info
            balance = info.get("balance", 0)
            currency = info.get("currency", "USD")
            is_demo = info.get("is_virtual", True)

            # ── Account summary ────────────────────────────────────────
            mode_tag = "DEMO" if is_demo else "LIVE"
            mode_color = "#58a6ff" if is_demo else "#f85149"
            st.markdown(f"""
            <div style="background:rgba(0,0,0,.3);border:1px solid {mode_color};
                 border-radius:10px;padding:14px 18px;margin-bottom:14px">
              <div style="display:flex;align-items:center;gap:12px">
                <span style="font-size:1.8rem">{'📘' if is_demo else '💰'}</span>
                <div>
                  <div style="color:{mode_color};font-size:1.1rem;font-weight:700">
                    {mode_tag} Account · {info.get('login_id','')}</div>
                  <div style="color:#e6edf3;font-size:1.3rem;font-weight:700">
                    {currency} {balance:,.2f}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            lp1, lp2, lp3, lp4 = st.columns(4)
            initial = 10_000.0
            pnl_live = balance - initial
            pnl_pct = (pnl_live / initial * 100) if initial > 0 else 0
            lp1.metric("Current Balance", f"{currency} {balance:,.2f}",
                       f"{pnl_pct:+.2f}%", delta_color="normal" if pnl_live >= 0 else "inverse")
            lp2.metric("Total P&L", f"${pnl_live:+,.2f}",
                       delta_color="normal" if pnl_live >= 0 else "inverse")
            lp3.metric("Account Type", mode_tag)
            lp4.metric("Email", info.get("email", "—")[:20])

            st.markdown("---")

            # ── Open positions ─────────────────────────────────────────
            st.markdown('<span class="sec">Open Positions</span>', unsafe_allow_html=True)
            client = st.session_state.deriv_client
            if client:
                try:
                    contracts = client.get_open_contracts()
                    if contracts:
                        df_pos = pd.DataFrame(contracts)
                        st.dataframe(df_pos, width="stretch", hide_index=True)
                        total_unrealized = sum(
                            float(c.get("profit", 0) or 0) for c in contracts
                        )
                        st.markdown(
                            f'<div style="color:{"#3fb950" if total_unrealized>=0 else "#f85149"};'
                            f'font-size:1rem;font-weight:700;margin:8px 0">'
                            f'Unrealized P&L: ${total_unrealized:+,.2f}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("No open positions")
                except Exception as e:
                    st.warning(f"Could not load open positions: {e}")

            st.markdown("---")

            # ── Recent trade history from Deriv ────────────────────────
            st.markdown('<span class="sec">Recent Trade History (Deriv)</span>', unsafe_allow_html=True)
            if client:
                try:
                    history = client.get_trade_history(30)
                    if history:
                        df_hist = pd.DataFrame(history)
                        st.dataframe(df_hist, width="stretch", hide_index=True)

                        # PnL chart from history
                        if "profit" in df_hist.columns:
                            pnls_hist = df_hist["profit"].astype(float).tolist()
                            cum_pnl = np.cumsum(pnls_hist).tolist()
                            fig_live = go.Figure()
                            fig_live.add_trace(go.Scatter(
                                x=list(range(1, len(cum_pnl) + 1)),
                                y=cum_pnl,
                                mode="lines+markers",
                                name="Cumulative PnL",
                                fill="tozeroy",
                                line=dict(
                                    color="#3fb950" if cum_pnl[-1] >= 0 else "#f85149",
                                    width=2
                                ),
                                fillcolor="rgba(63,185,80,0.1)" if cum_pnl[-1] >= 0
                                    else "rgba(248,81,73,0.1)",
                                marker=dict(size=5),
                            ))
                            fig_live.add_hline(y=0, line=dict(color="#30363d", width=1))
                            fig_live.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                                height=280,
                                margin=dict(l=6, r=6, t=10, b=30),
                                xaxis_title="Trade #",
                                yaxis_title="Cumulative PnL ($)",
                                yaxis=dict(gridcolor="#21262d"),
                                xaxis=dict(gridcolor="#21262d"),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_live, width="stretch")
                    else:
                        st.info("No trade history found yet. The bot needs to place and close trades first.")
                except Exception as e:
                    st.warning(f"Could not load trade history: {e}")

            # ── Paper trades performance ───────────────────────────────
            trades_paper = load_trades()
            if trades_paper:
                st.markdown("---")
                st.markdown('<span class="sec">Paper Trade Performance</span>', unsafe_allow_html=True)
                df_paper = pd.DataFrame(trades_paper)
                if "pnl" in df_paper.columns:
                    paper_pnls = df_paper["pnl"].astype(float).tolist()
                    paper_cum = np.cumsum(paper_pnls).tolist()
                    pp1, pp2, pp3, pp4 = st.columns(4)
                    pp1.metric("Paper Trades", len(df_paper))
                    pp2.metric("Paper Win Rate",
                               f"{(df_paper['pnl']>0).sum() / len(df_paper):.0%}")
                    pp3.metric("Paper PnL", f"${sum(paper_pnls):+,.2f}")
                    pp4.metric("Paper Balance",
                               f"${10_000 + sum(paper_pnls):,.2f}")
                    if len(paper_cum) > 1:
                        fig_paper = go.Figure()
                        fig_paper.add_trace(go.Scatter(
                            x=list(range(1, len(paper_cum) + 1)),
                            y=paper_cum,
                            mode="lines+markers",
                            fill="tozeroy",
                            line=dict(color="#a371f7", width=2),
                            fillcolor="rgba(163,113,247,0.1)",
                            marker=dict(size=4),
                        ))
                        fig_paper.add_hline(y=0, line=dict(color="#30363d", width=1))
                        fig_paper.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                            height=250,
                            margin=dict(l=6, r=6, t=10, b=30),
                            xaxis_title="Trade #", yaxis_title="Cumulative PnL ($)",
                            yaxis=dict(gridcolor="#21262d"),
                            xaxis=dict(gridcolor="#21262d"),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_paper, width="stretch")
        lot2=max(1.0,min(round(ra/sl_sz,2),500)) if sl_sz>0 else 1.0
        rc1,rc2,rc3,rc4=st.columns(4)
        rc1.metric("Risk Amount",f"${ra:.2f}")
        rc2.metric("SL Distance",fmt(symbol,sl_sz))
        rc3.metric("Stake / Lot",f"${lot2:.2f}")
        rc4.metric("TP Target",fmt(symbol,sl_sz*2))
        st.caption("For Deriv, 'Stake' = amount invested per trade. "
                   "Use Deriv multipliers to control your effective position size.")

    # ══ TELEGRAM ═══════════════════════════════════════════════════════════
    with t10:
        st.markdown("### 📲 Telegram Settings & Testing")

        tg_c1, tg_c2 = st.columns([1, 1])

        with tg_c1:
            st.markdown('<span class="sec">Configuration</span>', unsafe_allow_html=True)

            # Load current env values
            cur_token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
            cur_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

            tg_token = st.text_input(
                "Bot Token",
                value=cur_token,
                type="password",
                help="Get from @BotFather on Telegram",
                key="tg_token_input",
            )
            tg_chat_id = st.text_input(
                "Chat ID",
                value=cur_chat_id,
                help="Your Telegram user/group chat ID. Use @userinfobot to find it.",
                key="tg_chat_input",
            )

            # Save to .env
            if st.button("💾 Save Telegram Config", use_container_width=True):
                env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
                env_lines = []
                if os.path.exists(env_path):
                    with open(env_path) as ef:
                        env_lines = ef.readlines()

                # Update or add keys
                updated = {"TELEGRAM_BOT_TOKEN": False, "TELEGRAM_CHAT_ID": False}
                new_lines = []
                for line in env_lines:
                    stripped = line.strip()
                    if stripped.startswith("TELEGRAM_BOT_TOKEN="):
                        new_lines.append(f"TELEGRAM_BOT_TOKEN={tg_token}\n")
                        updated["TELEGRAM_BOT_TOKEN"] = True
                    elif stripped.startswith("TELEGRAM_CHAT_ID="):
                        new_lines.append(f"TELEGRAM_CHAT_ID={tg_chat_id}\n")
                        updated["TELEGRAM_CHAT_ID"] = True
                    else:
                        new_lines.append(line)
                if not updated["TELEGRAM_BOT_TOKEN"]:
                    new_lines.append(f"TELEGRAM_BOT_TOKEN={tg_token}\n")
                if not updated["TELEGRAM_CHAT_ID"]:
                    new_lines.append(f"TELEGRAM_CHAT_ID={tg_chat_id}\n")

                with open(env_path, "w") as ef:
                    ef.writelines(new_lines)

                # Also update environment for current session
                os.environ["TELEGRAM_BOT_TOKEN"] = tg_token
                os.environ["TELEGRAM_CHAT_ID"] = tg_chat_id
                st.success("✅ Config saved to .env")

            st.markdown("---")
            st.markdown('<span class="sec">Alert Toggles</span>', unsafe_allow_html=True)

            from config import TELEGRAM_CONFIG as _TC
            atc1, atc2 = st.columns(2)
            with atc1:
                al_signal = st.toggle("🔔 Signal Alerts", value=_TC["alerts"]["signal_gen"], key="tg_al_sig")
                al_open   = st.toggle("📈 Trade Open",    value=_TC["alerts"]["trade_open"], key="tg_al_open")
                al_close  = st.toggle("📉 Trade Close",   value=_TC["alerts"]["trade_close"], key="tg_al_close")
            with atc2:
                al_report = st.toggle("📊 Daily Report",  value=_TC["alerts"]["daily_report"], key="tg_al_rpt")
                al_kill   = st.toggle("🚨 Kill Switch",   value=_TC["alerts"]["kill_switch"], key="tg_al_kill")
                al_error  = st.toggle("❌ Errors",         value=_TC["alerts"]["error"], key="tg_al_err")

            # Apply toggle changes to config in-memory
            _TC["alerts"]["signal_gen"]   = al_signal
            _TC["alerts"]["trade_open"]   = al_open
            _TC["alerts"]["trade_close"]  = al_close
            _TC["alerts"]["daily_report"] = al_report
            _TC["alerts"]["kill_switch"]  = al_kill
            _TC["alerts"]["error"]        = al_error

        with tg_c2:
            st.markdown('<span class="sec">Connection Test</span>', unsafe_allow_html=True)

            # Status indicator
            eff_token = tg_token or cur_token
            eff_chat  = tg_chat_id or cur_chat_id
            has_creds = bool(eff_token and eff_chat)

            if has_creds:
                st.markdown(
                    '<div class="card">'
                    '🟢 <b>Credentials configured</b><br>'
                    f'<span style="color:#8b949e">Token: ...{eff_token[-8:] if len(eff_token) > 8 else "***"} '
                    f'| Chat: {eff_chat}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="card">'
                    '🔴 <b>Missing credentials</b><br>'
                    '<span style="color:#8b949e">Enter Bot Token and Chat ID on the left</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # Test buttons
            tbt1, tbt2 = st.columns(2)

            with tbt1:
                if st.button("📡 Test Connection", use_container_width=True, disabled=not has_creds):
                    try:
                        resp = requests.get(
                            f"https://api.telegram.org/bot{eff_token}/getMe",
                            timeout=10,
                        )
                        if resp.ok:
                            bot_info = resp.json().get("result", {})
                            bot_name = bot_info.get("first_name", "Unknown")
                            bot_user = bot_info.get("username", "?")
                            st.success(f"✅ Connected to **@{bot_user}** ({bot_name})")
                        else:
                            err = resp.json().get("description", resp.text)
                            st.error(f"❌ Connection failed: {err}")
                    except Exception as exc:
                        st.error(f"❌ Error: {exc}")

            with tbt2:
                if st.button("📤 Send Test Message", use_container_width=True, disabled=not has_creds):
                    try:
                        test_msg = (
                            "🤖 <b>AYH AI Trading Bot — Test</b>\n"
                            "─────────────────────\n"
                            f"✅ Connection OK!\n"
                            f"Chat ID: <code>{eff_chat}</code>\n"
                            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                            "─────────────────────\n"
                            "<i>Your Telegram notifications are working.</i>"
                        )
                        resp = requests.post(
                            f"https://api.telegram.org/bot{eff_token}/sendMessage",
                            json={"chat_id": eff_chat, "text": test_msg, "parse_mode": "HTML"},
                            timeout=10,
                        )
                        if resp.ok:
                            st.success("✅ Test message sent! Check your Telegram.")
                        else:
                            err = resp.json().get("description", resp.text)
                            st.error(f"❌ Send failed: {err}")
                    except Exception as exc:
                        st.error(f"❌ Error: {exc}")

            st.markdown("---")
            st.markdown('<span class="sec">Send Custom Signal</span>', unsafe_allow_html=True)

            with st.form("tg_test_signal", clear_on_submit=False):
                tsc1, tsc2 = st.columns(2)
                with tsc1:
                    test_sym = st.selectbox("Symbol", ALL_SYMBOLS, index=0, key="tg_test_sym")
                    test_dir = st.selectbox("Direction", ["BUY", "SELL"], key="tg_test_dir")
                with tsc2:
                    test_tf  = st.selectbox("Timeframe", TF_OPTIONS, index=3, key="tg_test_tf")
                    test_conf= st.slider("Confidence", 0.0, 1.0, 0.75, 0.05, key="tg_test_conf")

                test_entry = st.number_input("Entry Price", value=0.0, format="%.5f", key="tg_test_entry")
                test_sl    = st.number_input("Stop Loss",   value=0.0, format="%.5f", key="tg_test_sl")
                test_tp    = st.number_input("Take Profit", value=0.0, format="%.5f", key="tg_test_tp")

                submitted = st.form_submit_button(
                    "📲 Send Test Signal",
                    use_container_width=True,
                    disabled=not has_creds,
                )
                if submitted:
                    dir_emoji = "📈" if test_dir == "BUY" else "📉"
                    conf_bars = int(test_conf * 10)
                    conf_visual = "█" * conf_bars + "░" * (10 - conf_bars)

                    # Determine decimal places
                    dec = 5 if test_entry < 50 else 2 if test_entry > 0 else 5
                    entry_s = f"{test_entry:.{dec}f}" if test_entry else "—"
                    sl_s    = f"{test_sl:.{dec}f}"    if test_sl else "—"
                    tp_s    = f"{test_tp:.{dec}f}"    if test_tp else "—"
                    rr_s    = "—"
                    if test_sl and test_tp and test_entry:
                        sl_d = abs(test_entry - test_sl)
                        tp_d = abs(test_tp - test_entry)
                        rr_s = f"1:{tp_d/sl_d:.1f}" if sl_d > 0 else "—"

                    sig_msg = (
                        f"{dir_emoji} <b>SIGNAL: {test_sym} {test_tf}</b>\n"
                        f"{'─'*28}\n"
                        f"Direction  : <b>{test_dir}</b>\n"
                        f"Confidence : [{conf_visual}] {test_conf:.1%}\n"
                        f"\n"
                        f"💰 <b>Trade Levels</b>\n"
                        f"Entry : <code>{entry_s}</code>\n"
                        f"SL    : <code>{sl_s}</code>\n"
                        f"TP    : <code>{tp_s}</code>\n"
                        f"R:R   : <b>{rr_s}</b>\n"
                        f"{'─'*28}\n"
                        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n"
                        f"<i>⚠️ TEST SIGNAL — Not a real trade</i>"
                    )
                    try:
                        resp = requests.post(
                            f"https://api.telegram.org/bot{eff_token}/sendMessage",
                            json={"chat_id": eff_chat, "text": sig_msg, "parse_mode": "HTML"},
                            timeout=10,
                        )
                        if resp.ok:
                            st.success("✅ Test signal sent!")
                        else:
                            err = resp.json().get("description", resp.text)
                            st.error(f"❌ {err}")
                    except Exception as exc:
                        st.error(f"❌ {exc}")

            st.markdown("---")
            st.markdown('<span class="sec">Subscribers</span>', unsafe_allow_html=True)

            subs_file = os.path.join("logs", "subscribers.json")
            if os.path.exists(subs_file):
                try:
                    with open(subs_file) as sf:
                        subs_data = json.load(sf)
                    sub_list = subs_data.get("subscribers", [])
                    st.metric("Total Subscribers", len(sub_list) + (1 if eff_chat else 0))
                    if sub_list:
                        for sid in sub_list:
                            st.markdown(f"<code>{sid}</code>", unsafe_allow_html=True)
                    else:
                        st.caption("No additional subscribers. Users can DM /subscribe to your bot.")
                except Exception:
                    st.caption("Could not load subscriber data.")
            else:
                st.metric("Total Subscribers", 1 if eff_chat else 0)
                st.caption("No additional subscribers yet.")

    # ── Auto refresh ──────────────────────────────────────────────────────
    if auto_ref:
        ph=st.empty()
        for s in range(ref_s,0,-1): ph.caption(f"🔄 Refreshing in {s}s..."); time.sleep(1)
        st.cache_data.clear(); st.rerun()


if __name__=="__main__":
    main()
