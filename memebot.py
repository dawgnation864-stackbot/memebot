#!/usr/bin/env python3
"""
Memebot - Solana memecoin bot (cloud ready).
Educational example ONLY.
Not financial advice. Use at your own risk.

Default: SIMULATION_MODE = True
To go live:
  - Set SIMULATION_MODE=False in environment
  - Add WALLET_PRIVATE_KEY and WITHDRAWAL_ADDRESS
"""

from __future__ import annotations

# ---------- standard library ----------
import os
import sqlite3
import asyncio
import base64
import time
from datetime import datetime, date

# ---------- pip deps ----------
import requests
import numpy as np
import pandas as pd
import schedule
from dotenv import load_dotenv

# ---------- optional Solana (safe import) ----------
try:
    from solana.keypair import Keypair
    from solana.rpc.api import Client
    from solana.publickey import PublicKey

    SOLANA_OK = True
except:
    SOLANA_OK = False
    print("[solana] Solana libs not found, staying in simulation until installed.")

# ---------- load .env ----------
load_dotenv()

# ---------- config ----------
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "True").lower() == "true"
START_MODE = os.getenv("START_MODE", "start").lower()

STARTING_USD = float(os.getenv("STARTING_USD", "100"))
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "250"))
DAILY_LOSS_LIMIT_USD = float(os.getenv("DAILY_LOSS_LIMIT_USD", "25"))

TRADE_THRESHOLD = float(os.getenv("TRADE_THRESHOLD", "0.72"))

DB_FILE = os.getenv("DB_FILE", "memebot.db")

SOLANA_RPC = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com")
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "")
WITHDRAWAL_ADDRESS = os.getenv("WITHDRAWAL_ADDRESS", "")
PIN = os.getenv("PIN", "1234")

# ---------- runtime state ----------
STARTING_SOL = STARTING_USD / SOL_PRICE_USD
current_capital_sol = STARTING_SOL
today_pnl_sol = 0.0
today_trade_count = 0
today_date = date.today()

# ---------- DB ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
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
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY,
            date TEXT,
            capital_sol REAL,
            daily_pnl_sol REAL,
            trades_today INTEGER
        )
    """)

    conn.commit()
    conn.close()


def log_trade(ts, symbol, ca, action, size_sol, prob, rr, pnl_sol, mode):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO trades
        (ts, symbol, contract_address, action, size_sol, prob, risk_ratio, pnl_sol, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, symbol, ca, action, size_sol, prob, rr, pnl_sol, mode))
    conn.commit()
    conn.close()

# ---------- Solana wallet (only if libs present) ----------
def init_solana_wallet():
    if SIMULATION_MODE:
        print("[wallet] In simulation mode — no wallet needed.")
        return None, None

    if not SOLANA_OK:
        print("[wallet] Solana libs missing → simulation mode forced.")
        return None, None

    if not WALLET_PRIVATE_KEY:
        print("[wallet] WALLET_PRIVATE_KEY not provided.")
        return None, None

    try:
        from base58 import b58decode

        secret = b58decode(WALLET_PRIVATE_KEY)
        wallet = Keypair.from_secret_key(secret)
        client = Client(SOLANA_RPC)

        print(f"[wallet] Loaded wallet: {wallet.public_key}")
        return wallet, client
    except Exception as exc:
        print(f"[wallet] load error: {exc}")
        return None, None

# ---------- signal generator ----------
def fetch_signals():
    # stub: later replace with GMGN, Pumpfun, Twitter AI, etc
    return [
        ("DEMO", "So11111111111111111111111111111111111111112", 0.8, 0.1)
    ]

# ---------- simulate trade ----------
def simulate_trade(sol, prob, rr):
    size = sol
    ret = (0.862 if prob >= TRADE_THRESHOLD else -0.2)

    pnl = size * ret
    max_loss = DAILY_LOSS_LIMIT_USD / SOL_PRICE_USD
    if pnl < -max_loss:
        pnl = -max_loss

    new_cap = sol + pnl
    return new_cap, pnl

# ---------- core logic ----------
def analyze_and_trade(wallet, client):
    global current_capital_sol, today_pnl_sol, today_trade_count, today_date

    # new day → log summary
    if date.today() != today_date:
        print(f"[summary] {today_date} pnl={today_pnl_sol:.4f} SOL")
        today_date = date.today()
        today_pnl_sol = 0
        today_trade_count = 0

    print("[job] Fetching signals…")
    signals = fetch_signals()
    if not signals:
        print("[job] No signals.")
        return

    for sym, ca, prob, rr in signals:
        print(f"[signal] {sym} prob={prob:.3f} threshold={TRADE_THRESHOLD:.3f}")

        if prob < TRADE_THRESHOLD:
            print("[skip] Probability too low.")
            continue

        sol_before = current_capital_sol
        ts = datetime.utcnow()

        if wallet and client:
            print("[LIVE] Trading disabled placeholder (no Jupiter yet).")
            pnl = 0
            mode = "LIVE"
        else:
            new_cap, pnl = simulate_trade(sol_before, prob, rr)
            print(f"[SIM] traded {sol_before:.4f} → pnl={pnl:.4f} new={new_cap:.4f}")
            current_capital_sol = new_cap
            mode = "SIM"

        today_pnl_sol += pnl
        today_trade_count += 1

        log_trade(ts, sym, ca, "BUY", sol_before, prob, rr, pnl, mode)
        break  # one trade per cycle

# ---------- summary ----------
def daily_summary_job():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT COALESCE(SUM(pnl_sol),0), COUNT(*)
        FROM trades
        WHERE date(ts) = date('now','utc')
    """)
    total, num = c.fetchone()
    conn.close()

    print(f"[summary] trades={num} pnl={total:.4f} SOL (~${total*SOL_PRICE_USD:.2f})")

# ---------- main ----------
def main():
    print(f"[start] Memebot | SIM={SIMULATION_MODE} | START={STARTING_SOL:.4f} SOL")
    init_db()

    wallet, client = init_solana_wallet()

    if START_MODE == "withdraw":
        print("[withdraw] Not implemented yet.")
        return

    # run once immediately
    analyze_and_trade(wallet, client)

    # schedule
    schedule.every(1).hours.do(lambda: analyze_and_trade(wallet, client))
    schedule.every().day.at("23:59").do(daily_summary_job)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
