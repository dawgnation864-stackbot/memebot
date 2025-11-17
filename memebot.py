#!/usr/bin/env python3
"""
Memebot - Solana memecoin bot (Railway-ready).

⚠️ Educational example ONLY. Not financial advice.
Extreme risk. Never trade money you cannot afford to lose.

- Default: SIMULATION_MODE = False  (live mode intent)
- To TEST ONLY, you can set SIMULATION_MODE=True in Railway variables.
- For live mode you MUST set:
    WALLET_PRIVATE_KEY  (base58-encoded secret key)
    WITHDRAWAL_ADDRESS  (your Solana address)
"""

from __future__ import annotations

# ---------- standard library ----------
import os
import sqlite3
import asyncio
import base64
import time
from datetime import datetime, date

# ---------- third-party ----------
import requests
import numpy as np
import pandas as pd
import schedule
from dotenv import load_dotenv
from duckduckgo_search import DDGS  # placeholder for future signal sources

# ---------- Solana / Jupiter stack ----------
try:
    # Core Solana client & types
    from solana.rpc.api import Client
    from solana.rpc.types import TxOpts
    from solana.keypair import Keypair
    from solana.publickey import PublicKey
    from solana.transaction import VersionedTransaction
    from solana.system_program import TransferParams, transfer

    SOLANA_OK = True
except ImportError as exc:
    SOLANA_OK = False
    print(f"[solana] Import error: {exc!r}")
    # We do NOT force simulation here; we just won't be able to go live.

# ---------- load .env ----------
load_dotenv()

# ---------- configuration from env ----------
# Default to LIVE intent (False). You can override in Railway Variables.
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "False").lower() == "true"
START_MODE = os.getenv("START_MODE", "start").lower()  # 'start' or 'withdraw'

STARTING_USD = float(os.getenv("STARTING_USD", "100"))
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "250"))
DAILY_LOSS_LIMIT_USD = float(os.getenv("DAILY_LOSS_LIMIT_USD", "25"))

TRADE_THRESHOLD = float(os.getenv("TRADE_THRESHOLD", "0.72"))

DB_FILE = os.getenv("DB_FILE", "memebot.db")

JUPITER_ENDPOINT = os.getenv(
    "JUPITER_ENDPOINT", "https://quote-api.jup.ag/v6"
)
SOLANA_RPC = os.getenv(
    "SOLANA_RPC", "https://api.mainnet-beta.solana.com"
)
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "")
WITHDRAWAL_ADDRESS = os.getenv("WITHDRAWAL_ADDRESS", "")
PIN = os.getenv("PIN", "1234")

# ---------- simple in-memory state ----------
STARTING_SOL = STARTING_USD / SOL_PRICE_USD
current_capital_sol = STARTING_SOL
today_pnl_sol = 0.0
today_trade_count = 0
today_date = date.today()

# ============================================================
#   DATABASE HELPERS
# ============================================================


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TIMESTAMP,
            symbol TEXT,
            contract_address TEXT,
            action TEXT,
            size_sol REAL,
            prob REAL,
            risk_ratio REAL,
            pnl_sol REAL,
            mode TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            capital_sol REAL,
            daily_pnl_sol REAL,
            trades_today INTEGER
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS learn_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            symbol TEXT,
            contract_address TEXT,
            prob REAL,
            risk_ratio REAL,
            trade_size_sol REAL,
            pnl_sol REAL,
            label INTEGER
        )
        """
    )

    conn.commit()
    conn.close()


def log_trade(
    ts: datetime,
    symbol: str,
    ca: str,
    action: str,
    size_sol: float,
    prob: float,
    risk_ratio: float,
    pnl_sol: float,
    mode: str,
) -> None:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO trades
        (ts, symbol, contract_address, action, size_sol, prob,
         risk_ratio, pnl_sol, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, symbol, ca, action, size_sol, prob, risk_ratio, pnl_sol, mode),
    )
    conn.commit()
    conn.close()


def log_portfolio_for_day(
    day: date, capital_sol: float, daily_pnl_sol: float, trades_today: int
) -> None:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO portfolio (date, capital_sol, daily_pnl_sol, trades_today)
        VALUES (?, ?, ?, ?)
        """,
        (day.isoformat(), capital_sol, daily_pnl_sol, trades_today),
    )
    conn.commit()
    conn.close()


def insert_learn_event(
    ts: datetime,
    symbol: str,
    ca: str,
    prob: float,
    risk_ratio: float,
    trade_size_sol: float,
    pnl_sol: float,
) -> None:
    """Store result of a trade for simple 'learning'."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    label = 1 if pnl_sol > 0 else 0

    c.execute(
        """
        INSERT INTO learn_events (
            timestamp, symbol, contract_address, prob, risk_ratio,
            trade_size_sol, pnl_sol, label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts,
            symbol,
            ca,
            float(prob),
            float(risk_ratio),
            float(trade_size_sol),
            float(pnl_sol),
            int(label),
        ),
    )
    conn.commit()
    conn.close()

# ============================================================
#   SIMPLE LEARNING
# ============================================================


def get_recent_stats() -> dict:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        SELECT prob, pnl_sol FROM learn_events
        WHERE timestamp >= datetime('now', '-3 days')
        """
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {"count": 0, "win_rate": 0.6, "avg_prob": 0.75}

    probs = np.array([r[0] for r in rows], dtype=float)
    pnls = np.array([r[1] for r in rows], dtype=float)
    wins = (pnls > 0).astype(float)

    win_rate = float(wins.mean())
    avg_prob = float(probs.mean())
    return {"count": len(rows), "win_rate": win_rate, "avg_prob": avg_prob}


def train_model_stub() -> None:
    """Update TRADE_THRESHOLD based on recent performance."""
    global TRADE_THRESHOLD
    stats = get_recent_stats()
    if stats["count"] == 0:
        print("[learn] Not enough data yet.")
        return

    base = 0.7
    adjust = (0.5 - stats["win_rate"]) * 0.1
    new_threshold = min(0.9, max(0.6, base + adjust))

    print(
        f"[learn] count={stats['count']} win_rate={stats['win_rate']:.2f} "
        f"old_threshold={TRADE_THRESHOLD:.3f} "
        f"new_threshold={new_threshold:.3f}"
    )
    TRADE_THRESHOLD = new_threshold

# ============================================================
#   SOLANA / JUPITER HELPERS
# ============================================================


def init_solana_wallet():
    """
    Load wallet from WALLET_PRIVATE_KEY if libs and key are available.

    In LIVE mode we require:
      - SOLANA_OK = True
      - WALLET_PRIVATE_KEY set
    Otherwise we return (None, None) and live trades are skipped.
    """
    if not SOLANA_OK:
        print("[wallet] Solana Python libs are not available.")
        return None, None

    if not WALLET_PRIVATE_KEY:
        print("[wallet] WALLET_PRIVATE_KEY not set, cannot trade live.")
        return None, None

    try:
        import base58

        secret_key_bytes = base58.b58decode(WALLET_PRIVATE_KEY)
        wallet = Keypair.from_secret_key(secret_key_bytes)
        client = Client(SOLANA_RPC)
        print(f"[wallet] Loaded wallet: {wallet.public_key}")
        return wallet, client
    except Exception as exc:
        print(f"[wallet] Error loading wallet: {exc}")
        return None, None


def jupiter_quote(input_mint: str, output_mint: str, amount: int) -> dict:
    url = f"{JUPITER_ENDPOINT}/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": amount,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


async def _execute_swap(quote: dict, wallet: Keypair, client: Client) -> str | None:
    """
    Execute a Jupiter swap transaction.

    Wrapped in try/except so that any incompatibility with solana library
    just logs an error instead of crashing the container.
    """
    try:
        swap_url = f"{JUPITER_ENDPOINT}/swap"
        payload = {
            "quoteResponse": quote,
            "userPublicKey": str(wallet.public_key),
            "wrapAndUnwrapSol": True,
            "prioritizationFeeLamports": "auto",
        }

        r = requests.post(swap_url, json=payload, timeout=30)
        r.raise_for_status()
        swap_tx_b64 = r.json()["swapTransaction"]

        swap_tx_bytes = base64.b64decode(swap_tx_b64)
        tx = VersionedTransaction.deserialize(swap_tx_bytes)
        tx.sign([wallet])

        sig = client.send_raw_transaction(
            tx.serialize(),
            opts=TxOpts(skip_preflight=True, max_retries=3),
        )
        print(f"[swap] sent tx: {sig}")
        return str(sig)
    except Exception as exc:
        print(f"[swap] error while building/sending tx: {exc}")
        return None


def place_live_swap(
    wallet: Keypair,
    client: Client,
    token_mint: str,
    amount_lamports: int,
    is_buy: bool,
) -> str | None:
    """Execute a live swap using Jupiter."""
    try:
        sol_mint = "So11111111111111111111111111111111111111112"
        input_mint = sol_mint if is_buy else token_mint
        output_mint = token_mint if is_buy else sol_mint

        quote = jupiter_quote(input_mint, output_mint, amount_lamports)
        sig = asyncio.run(_execute_swap(quote, wallet, client))
        if sig:
            print(
                f"[swap] {'BUY' if is_buy else 'SELL'} "
                f"{amount_lamports / 1e9:.4f} SOL of token {token_mint}"
            )
        return sig
    except Exception as exc:
        print(f"[swap] error: {exc}")
        return None

# ---------- emergency withdrawal ----------


def emergency_withdraw(wallet: Keypair, client: Client) -> None:
    """Transfer almost all SOL to WITHDRAWAL_ADDRESS after PIN confirmation."""
    if not wallet or not client:
        print("[withdraw] No wallet/client; cannot withdraw.")
        return

    pin_in = input("Enter PIN to withdraw and stop bot: ").strip()
    if pin_in != PIN:
        print("[withdraw] Wrong PIN.")
        return

    if not WITHDRAWAL_ADDRESS:
        print("[withdraw] WITHDRAWAL_ADDRESS not set.")
        return

    try:
        bal_resp = client.get_balance(wallet.public_key)
        lamports = bal_resp["result"]["value"]
        print(f"[withdraw] balance = {lamports / 1e9:.4f} SOL")

        send_lamports = int(max(0, lamports - int(0.003 * 1e9)))  # leave fees

        params = TransferParams(
            from_pubkey=wallet.public_key,
            to_pubkey=PublicKey(WITHDRAWAL_ADDRESS),
            lamports=send_lamports,
        )
        ix = transfer(params)
        recent = client.get_latest_blockhash()["result"]["value"]["blockhash"]

        tx = VersionedTransaction(
            message=ix.to_message(recent_blockhash=recent),
            signatures=[wallet],
        )
        sig = client.send_raw_transaction(
            tx.serialize(), opts=TxOpts(skip_preflight=True)
        )
        print(
            f"[withdraw] sent {send_lamports / 1e9:.4f} SOL "
            f"→ {WITHDRAWAL_ADDRESS} (sig={sig})"
        )
    except Exception as exc:
        print(f"[withdraw] error: {exc}")

# ============================================================
#   SIGNAL GENERATION (STUB)
# ============================================================


def fetch_signals() -> list[tuple[str, str, float, float]]:
    """
    Return a list of (symbol, contract_address, prob, risk_ratio).

    Right now this is a fake single signal so your pipeline runs.
    You can later plug in GMGN / Twitter / PumpFun scanners here.
    """
    symbol = "DEMO"
    ca = "So11111111111111111111111111111111111111112"  # SOL mint as dummy
    prob = 0.8
    risk_ratio = 0.1
    return [(symbol, ca, prob, risk_ratio)]

# ============================================================
#   TRADING CORE
# ============================================================


def simulate_trade(
    current_sol: float, prob: float, risk_ratio: float
) -> tuple[float, float]:
    """
    Very simple challenge simulation:
    - if prob >= threshold: +86.2% on trade size
    - else: -20%
    - single-trade loss is capped by DAILY_LOSS_LIMIT_USD
    """
    trade_size_sol = current_sol  # all-in
    if prob >= TRADE_THRESHOLD:
        ret = 0.862
    else:
        ret = -0.2

    pnl_sol = trade_size_sol * ret

    # Cap loss
    max_loss_sol = DAILY_LOSS_LIMIT_USD / SOL_PRICE_USD
    if pnl_sol < -max_loss_sol:
        pnl_sol = -max_loss_sol

    new_capital = current_sol + pnl_sol
    return new_capital, pnl_sol


def analyze_and_trade(wallet: Keypair | None, client: Client | None) -> None:
    """Main per-cycle logic: get signals, maybe trade, update DB and stats."""
    global current_capital_sol, today_pnl_sol, today_trade_count, today_date

    # Day rollover & summary logging
    if date.today() != today_date:
        log_portfolio_for_day(
            today_date, current_capital_sol, today_pnl_sol, today_trade_count
        )
        print(
            f"[summary] {today_date.isoformat()} "
            f"pnl={today_pnl_sol:.4f} SOL trades={today_trade_count}"
        )
        today_date = date.today()
        today_pnl_sol = 0.0
        today_trade_count = 0

    print("[job] Fetching signals...")
    signals = fetch_signals()
    if not signals:
        print("[job] No signals this cycle.")
        return

    for symbol, ca, prob, risk_ratio in signals:
        print(
            f"[signal] {symbol} prob={prob:.3f} "
            f"threshold={TRADE_THRESHOLD:.3f}"
        )

        if prob < TRADE_THRESHOLD:
            print("[decision] Skip (probability below threshold).")
            continue

        trade_size_sol = current_capital_sol
        if trade_size_sol <=
