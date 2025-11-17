#!/usr/bin/env python3
"""
Memebot - Solana memecoin bot (cloud ready).

Educational example ONLY.
Not financial advice. Extremely high risk. Never trade money you cannot lose.

To trade for real you must:
- set SIMULATION_MODE=False in environment variables
- set WALLET_PRIVATE_KEY (base58) and WITHDRAWAL_ADDRESS
"""

from __future__ import annotations

# ---------- standard library imports ----------
import os
import sqlite3
import asyncio
import base64
import time
from datetime import datetime, date

# ---------- third-party imports ----------
import requests
import numpy as np
import pandas as pd
import schedule
from dotenv import load_dotenv
from duckduckgo_search import DDGS  # not used yet but kept for future

# ---------- optional Solana / Jupiter support ----------
SOLANA_OK = False
try:
    # solana-py client + system program
    from solana.rpc.api import Client as RpcClient
    from solana.publickey import PublicKey
    from solana.system_program import TransferParams, transfer

    # solders types for newer Jupiter tx format
    from solders.keypair import Keypair
    from solders.transaction import VersionedTransaction
    from solders.message import MessageV0
    from solders.instruction import Instruction, AccountMeta
    from solders.pubkey import Pubkey as SoldersPubkey

    SOLANA_OK = True
except Exception as exc:
    print(f"[solana] Import error: {exc!r}")
    SOLANA_OK = False

# ---------- load .env ----------
load_dotenv()

# ---------- configuration from env ----------
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "True").lower() == "true"
START_MODE = os.getenv("START_MODE", "start").lower()  # 'start' or 'withdraw'

STARTING_USD = float(os.getenv("STARTING_USD", "100"))
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "250"))
DAILY_LOSS_LIMIT_USD = float(os.getenv("DAILY_LOSS_LIMIT_USD", "25"))

TRADE_THRESHOLD = float(os.getenv("TRADE_THRESHOLD", "0.72"))

DB_FILE = os.getenv("DB_FILE", "memebot.db")

JUPITER_ENDPOINT = os.getenv("JUPITER_ENDPOINT", "https://quote-api.jup.ag/v6")
JUPITER_ENDPOINT_FALLBACK = os.getenv("JUPITER_ENDPOINT_FALLBACK", "https://api.jup.ag")

SOLANA_RPC = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com")
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "")
WITHDRAWAL_ADDRESS = os.getenv("WITHDRAWAL_ADDRESS", "")
PIN = os.getenv("PIN", "1234")

# ---------- simple in-memory state ----------
STARTING_SOL = STARTING_USD / SOL_PRICE_USD
current_capital_sol = STARTING_SOL
today_pnl_sol = 0.0
today_trade_count = 0
today_date = date.today()

# ---------- database helpers ----------


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
        INSERT INTO trades (ts, symbol, contract_address, action, size_sol, prob,
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
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    label = 1 if pnl_sol > 0 else 0
    query = """
    INSERT INTO learn_events (
        timestamp, symbol, contract_address, prob, risk_ratio,
        trade_size_sol, pnl_sol, label
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    c.execute(
        query,
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


# ---------- very simple 'learning' ----------


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
    """Update threshold based on recent performance (very simple)."""
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
        f"old_threshold={TRADE_THRESHOLD:.3f} new_threshold={new_threshold:.3f}"
    )
    TRADE_THRESHOLD = new_threshold


# ---------- Solana / Jupiter helpers ----------


def init_solana_wallet():
    """Load wallet from WALLET_PRIVATE_KEY if libs and key are available."""
    if not SOLANA_OK:
        print("[wallet] SOLANA_OK=False (solders import failed).")
        return None, None

    if not WALLET_PRIVATE_KEY:
        print("[wallet] WALLET_PRIVATE_KEY not set; cannot go live.")
        return None, None

    try:
        import base58 as _b58

        secret_key_bytes = _b58.b58decode(WALLET_PRIVATE_KEY)
        wallet = Keypair.from_secret_key(secret_key_bytes)
        client = RpcClient(SOLANA_RPC)
        print(f"[wallet] Loaded wallet: {wallet.public_key}")
        return wallet, client
    except Exception as exc:
        print(f"[wallet] Error loading wallet: {exc}")
        return None, None


def _jupiter_request(path: str, params: dict | None = None) -> dict:
    """Call Jupiter with primary endpoint, then fallback if DNS fails."""
    url = f"{JUPITER_ENDPOINT.rstrip('/')}/{path.lstrip('/')}"
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        msg = repr(exc)
        print(f"[jupiter] primary endpoint error: {msg}")
        # DNS / hostname issues → try fallback host
        if "NameResolutionError" in msg or "No address associated with hostname" in msg:
            fb_url = f"{JUPITER_ENDPOINT_FALLBACK.rstrip('/')}/{path.lstrip('/')}"
            print(f"[jupiter] trying fallback endpoint: {fb_url}")
            try:
                r2 = requests.get(fb_url, params=params, timeout=20)
                r2.raise_for_status()
                return r2.json()
            except Exception as exc2:
                print(f"[jupiter] fallback endpoint error: {exc2!r}")
                raise
        raise


def jupiter_quote(input_mint: str, output_mint: str, amount: int) -> dict:
    params = {"inputMint": input_mint, "outputMint": output_mint, "amount": amount}
    return _jupiter_request("quote", params=params)


async def _execute_swap(quote: dict, wallet, client) -> str:
    swap_url = f"{JUPITER_ENDPOINT.rstrip('/')}/swap"
    payload = {
        "quoteResponse": quote,
        "userPublicKey": str(wallet.public_key),
        "wrapAndUnwrapSol": True,
        "prioritizationFeeLamports": "auto",
    }
    r = requests.post(swap_url, json=payload, timeout=20)
    r.raise_for_status()

    swap_tx_buf = base64.b64decode(r.json()["swapTransaction"])
    tx = VersionedTransaction.from_bytes(swap_tx_buf)
    tx.sign([wallet])
    sig = client.send_raw_transaction(bytes(tx))
    print(f"[swap] sent tx: {sig.value}")
    client.confirm_transaction(sig.value, commitment="confirmed")
    return sig.value


def place_live_swap(
    wallet, client, token_mint: str, amount_lamports: int, is_buy: bool
) -> str | None:
    """Execute a live swap using Jupiter."""
    try:
        sol_mint = "So11111111111111111111111111111111111111112"
        input_mint = sol_mint if is_buy else token_mint
        output_mint = token_mint if is_buy else sol_mint
        quote = jupiter_quote(input_mint, output_mint, amount_lamports)
        sig = asyncio.run(_execute_swap(quote, wallet, client))
        print(
            f"[swap] {'BUY' if is_buy else 'SELL'} {amount_lamports / 1e9:.4f} SOL "
            f"of token {token_mint}"
        )
        return sig
    except Exception as exc:
        print(f"[swap] error: {exc!r}")
        return None


# ---------- emergency withdrawal ----------


def emergency_withdraw(wallet, client) -> None:
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
        lamports = bal_resp.value
        print(f"[withdraw] balance = {lamports / 1e9:.4f} SOL")

        send_lamports = int(max(0, lamports - int(0.003 * 1e9)))  # leave fees

        params = TransferParams(
            from_pubkey=wallet.public_key,
            to_pubkey=Pubkey.from_string(WITHDRAWAL_ADDRESS),
            lamports=send_lamports,
        )
        ix = transfer(params)
        recent = client.get_latest_blockhash().value.blockhash
        msg = MessageV0.try_compile(
            payer=wallet.public_key,
            instructions=[ix],
            address_lookup_table_accounts=[],
            recent_blockhash=recent,
        )
        tx = VersionedTransaction(msg, [wallet])
        sig = client.send_raw_transaction(bytes(tx))
        client.confirm_transaction(sig.value)
        print(
            f"[withdraw] sent {send_lamports / 1e9:.4f} SOL → {WITHDRAWAL_ADDRESS}"
        )
    except Exception as exc:
        print(f"[withdraw] error: {exc!r}")


# ---------- signal generation (stub) ----------


def fetch_signals() -> list[tuple[str, str, float, float]]:
    """
    Return a list of (symbol, contract_address, prob, risk_ratio).

    For now we just return one fake signal so the logic works end-to-end.
    """
    symbol = "DEMO"
    ca = "So11111111111111111111111111111111111111112"
    prob = 0.8
    risk_ratio = 0.1
    return [(symbol, ca, prob, risk_ratio)]


# ---------- trading core ----------


def simulate_trade(
    current_sol: float, prob: float, risk_ratio: float
) -> tuple[float, float]:
    trade_size_sol = current_sol  # all-in for challenge
    if prob >= TRADE_THRESHOLD:
        ret = 0.862
    else:
        ret = -0.2

    pnl_sol = trade_size_sol * ret

    max_loss_sol = DAILY_LOSS_LIMIT_USD / SOL_PRICE_USD
    if pnl_sol < -max_loss_sol:
        pnl_sol = -max_loss_sol

    new_capital = current_sol + pnl_sol
    return new_capital, pnl_sol


def analyze_and_trade(wallet, client) -> None:
    global current_capital_sol, today_pnl_sol, today_trade_count, today_date

    # new day summary
    if date.today() != today_date:
        log_portfolio_for_day(
            today_date, current_capital_sol, today_pnl_sol, today_trade_count
        )
        print(
            f"[summary] {today_date.isoformat()} pnl={today_pnl_sol:.4f} "
            f"SOL trades={today_trade_count}"
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
            f"[signal] {symbol} ca={ca[:8]}... prob={prob:.3f} "
            f"risk_ratio={risk_ratio:.3f} threshold={TRADE_THRESHOLD:.3f}"
        )

        if prob < TRADE_THRESHOLD:
            print("[decision] Skip (probability below threshold).")
            continue

        trade_size_sol = current_capital_sol
        if trade_size_sol <= 0:
            print("[decision] No capital left; skipping.")
            continue

        ts = datetime.utcnow()

        if SIMULATION_MODE or not wallet or not client:
            new_capital, pnl_sol = simulate_trade(current_capital_sol, prob, risk_ratio)
            print(
                f"[SIM] traded {trade_size_sol:.4f} SOL, pnl={pnl_sol:.4f} "
                f"SOL → new_capital={new_capital:.4f} SOL"
            )
            mode = "SIM"
            current_capital_sol = new_capital
        else:
            lamports = int(trade_size_sol * 1e9)
            sig = place_live_swap(wallet, client, ca, lamports, is_buy=True)
            mode = "LIVE"
            if sig is None:
                pnl_sol = 0.0
                print("[LIVE] swap failed (no tx); treating pnl as 0.")
            else:
                pnl_sol = 0.0  # we’re not marking real PnL here
                print(f"[LIVE] swap tx={sig}, size={trade_size_sol:.4f} SOL")

        today_pnl_sol += pnl_sol
        today_trade_count += 1

        log_trade(
            ts,
            symbol,
            ca,
            "BUY",
            trade_size_sol,
            prob,
            risk_ratio,
            pnl_sol,
            mode,
        )
        insert_learn_event(ts, symbol, ca, prob, risk_ratio, trade_size_sol, pnl_sol)

        break  # only 1 trade per cycle


# ---------- daily summary & learning ----------


def daily_summary_job() -> None:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        SELECT COALESCE(SUM(pnl_sol), 0), COUNT(*) FROM trades
        WHERE date(ts) = date('now', 'utc')
        """
    )
    row = c.fetchone()
    conn.close()
    total_pnl, count = row if row else (0.0, 0)

    print(
        f"[summary] Today so far: trades={count} total_pnl={total_pnl:.4f} SOL "
        f"(~${total_pnl * SOL_PRICE_USD:.2f})"
    )


def daily_learning_job() -> None:
    print("[learn] Running daily learning job...")
    train_model_stub()


# ---------- main loop ----------


def main() -> None:
    print(
        f"[start] Memebot | SIM={SIMULATION_MODE} | START={STARTING_SOL:.4f} SOL"
    )
    init_db()

    wallet = None
    client = None

    if not SIMULATION_MODE:
        if not SOLANA_OK:
            print("[fatal] SOLANA_OK=False but SIMULATION_MODE=False; cannot go live.")
            return
        wallet, client = init_solana_wallet()
        if not wallet or not client:
            print("[fatal] SIMULATION_MODE=False but wallet/client not available.")
            return
    else:
        print("[start] Simulation mode ON.")

    if START_MODE == "withdraw":
        if wallet and client:
            emergency_withdraw(wallet, client)
        else:
            print("[withdraw] Cannot withdraw (no wallet/client).")
        return

    # schedule jobs
    schedule.every(1).hours.do(lambda: analyze_and_trade(wallet, client))
    schedule.every().day.at("23:59").do(daily_summary_job)
    schedule.every().day.at("02:00").do(daily_learning_job)

    # run one cycle immediately
    print(">>> memebot.py started (top of file reached)")
    analyze_and_trade(wallet, client)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
