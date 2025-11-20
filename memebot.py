"""
Memebot – Solana memecoin trading bot (Railway version)

Environment variables (set in Railway):

# Core switches
SIMULATION_MODE        -> "True" or "False"
START_MODE             -> "start"  (or "withdraw" for emergency withdraw)

# Wallet / RPC
WALLET_PRIVATE_KEY     -> base58-encoded Solana private key (NOT your seed phrase)
SOLANA_RPC             -> RPC URL (e.g. https://api.mainnet-beta.solana.com)
WITHDRAWAL_ADDRESS     -> (optional) address for emergency withdraw

# Starting balance / risk
STARTING_SOL           -> e.g. 0.4000
STARTING_USD           -> e.g. 100.0      (optional, for PnL display)
SOL_PRICE_USD          -> e.g. 200.0      (used if STARTING_USD is missing)
DAILY_LOSS_LIMIT_USD   -> e.g. 25.0
MAX_TRADE_RISK_SOL     -> e.g. 0.40       (max SOL per trade)

# Strategy thresholds
MIN_PROBABILITY        -> e.g. 0.72       (signal threshold)
STOP_LOSS_MULTIPLIER   -> e.g. 0.50
TAKE_PROFIT_MULTIPLIER -> e.g. 2.00

# Timing
SCAN_INTERVAL_SECONDS  -> e.g. 60

# Jupiter config
JUPITER_API_KEY        -> your Jupiter Ultra API key
JUPITER_API_BASE       -> your endpoint, e.g. https://api.jup.ag/ultra

# Safety
PIN                    -> 4+ digit pin you know (for future manual actions)
"""

from __future__ import annotations

# ---------- standard library imports ----------
import os
import sqlite3
import asyncio
import base64
import random
import time
from datetime import datetime, date
from typing import Optional, Dict, Any, Tuple

# ---------- third-party imports ----------
import requests
import schedule
from dotenv import load_dotenv
from base58 import b58decode

# ---------- optional Solana stack ----------
SOLANA_OK = False
try:
    from solana.keypair import Keypair
    from solana.publickey import PublicKey
    from solana.rpc.api import Client as SolClient
    from solana.transaction import Transaction
    from solana.system_program import TransferParams, transfer

    SOLANA_OK = True
except Exception as exc:
    print(f"[solana] Import error: {exc!r}")
    SOLANA_OK = False

# ---------- load .env (for local dev; Railway uses env directly) ----------
load_dotenv()

# ---------- configuration from env ----------
def env_bool(name: str, default: str = "False") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y")

SIMULATION_MODE       = env_bool("SIMULATION_MODE", "True")
START_MODE            = os.getenv("START_MODE", "start").strip().lower()

WALLET_PRIVATE_KEY    = os.getenv("WALLET_PRIVATE_KEY", "").strip()
SOLANA_RPC            = os.getenv("SOLANA_RPC", "").strip()
WITHDRAWAL_ADDRESS    = os.getenv("WITHDRAWAL_ADDRESS", "").strip()

STARTING_SOL          = float(os.getenv("STARTING_SOL", "0.40"))
STARTING_USD_ENV      = os.getenv("STARTING_USD", "").strip()
SOL_PRICE_USD         = float(os.getenv("SOL_PRICE_USD", "200.0"))

if STARTING_USD_ENV:
    STARTING_USD = float(STARTING_USD_ENV)
else:
    STARTING_USD = STARTING_SOL * SOL_PRICE_USD

DAILY_LOSS_LIMIT_USD  = float(os.getenv("DAILY_LOSS_LIMIT_USD", "25.0"))
MAX_TRADE_RISK_SOL    = float(os.getenv("MAX_TRADE_RISK_SOL", "0.40"))

MIN_PROBABILITY       = float(os.getenv("MIN_PROBABILITY", "0.72"))
STOP_LOSS_MULTIPLIER  = float(os.getenv("STOP_LOSS_MULTIPLIER", "0.50"))
TAKE_PROFIT_MULTIPLIER= float(os.getenv("TAKE_PROFIT_MULTIPLIER", "2.00"))

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))

JUPITER_API_KEY       = os.getenv("JUPITER_API_KEY", "").strip()
JUPITER_API_BASE      = os.getenv("JUPITER_API_BASE", "https://api.jup.ag").rstrip("/")

PIN                   = os.getenv("PIN", "").strip()

# ---------- constants ----------
SOL_MINT = "So11111111111111111111111111111111111111112"

# Aggressive meme token list.
# These are mint addresses that the bot can randomly choose from.
# You can expand this list using mints from GMGN / Jupiter token list.
MEME_TOKENS = [
    # These three are example mints you've already seen in your logs:
    {"symbol": "MEME1", "mint": "F5h1wNSMjU3mV8HqbWtpNQco6WcmnZuhA7gY2kLD3vYX"},
    {"symbol": "MEME2", "mint": "DjQHjXbVWvn1ddM5HE6sDW2iqQeu2C1DZJ5MJXk6w8Dz"},
    {"symbol": "MEME3", "mint": "FULdBJ8UfDxaTfb4qxwUeMb55q1P6p8v7uxxY4PnKbeC"},
    # Add more here once you have their mint addresses.
]

# ---------- small state ----------
db_path = "memebot_state.sqlite3"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS pnl (
        d TEXT PRIMARY KEY,
        start_usd REAL,
        realized_usd REAL
    )
    """
)
conn.commit()

# ---------- helpers for PnL / daily guard ----------
def today_str() -> str:
    return date.today().isoformat()

def load_today_pnl() -> Tuple[float, float]:
    cur.execute("SELECT start_usd, realized_usd FROM pnl WHERE d = ?", (today_str(),))
    row = cur.fetchone()
    if row:
        return row[0], row[1]
    # first run of the day
    cur.execute(
        "INSERT OR REPLACE INTO pnl (d, start_usd, realized_usd) VALUES (?, ?, ?)",
        (today_str(), STARTING_USD, 0.0),
    )
    conn.commit()
    return STARTING_USD, 0.0

def update_realized_pnl(delta_usd: float) -> float:
    start_usd, realized = load_today_pnl()
    realized += delta_usd
    cur.execute(
        "UPDATE pnl SET realized_usd = ? WHERE d = ?",
        (realized, today_str())
    )
    conn.commit()
    return realized

def daily_loss_exceeded() -> bool:
    start_usd, realized = load_today_pnl()
    loss = -realized if realized < 0 else 0.0
    if loss >= DAILY_LOSS_LIMIT_USD:
        print(
            f"[risk] Daily loss cap hit: loss={loss:.2f} USD "
            f"(limit {DAILY_LOSS_LIMIT_USD:.2f}). No more trades today."
        )
        return True
    return False

# ---------- Solana wallet / RPC ----------
def init_solana_wallet() -> Tuple[Optional[Keypair], Optional[SolClient]]:
    if not SOLANA_OK:
        print("[wallet] SOLANA_OK=False; solana libs not available.")
        return None, None
    if not WALLET_PRIVATE_KEY:
        print("[wallet] WALLET_PRIVATE_KEY not set.")
        return None, None
    if not SOLANA_RPC:
        print("[wallet] SOLANA_RPC not set.")
        return None, None

    try:
        secret_key_bytes = b58decode(WALLET_PRIVATE_KEY)
        wallet = Keypair.from_secret_key(secret_key_bytes)
        client = SolClient(SOLANA_RPC)
        print(f"[wallet] Loaded wallet: {str(wallet.public_key)}")
        return wallet, client
    except Exception as exc:
        print(f"[wallet] Error loading wallet: {exc!r}")
        return None, None

# ---------- Jupiter helpers ----------
def jupiter_request(path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generic helper for Jupiter Ultra API requests.
    Automatically injects Authorization header.
    """
    base = JUPITER_API_BASE.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    url = f"{base}{path}"

    headers = {
        "Content-Type": "application/json",
    }
    if JUPITER_API_KEY:
        headers["Authorization"] = f"Bearer {JUPITER_API_KEY}"

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[swap] quote error: {exc!r}")
        return None

def get_jupiter_quote(input_mint: str, output_mint: str, amount_lamports: int) -> Optional[Dict[str, Any]]:
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": amount_lamports,
        "slippageBps": 500,
        "onlyDirectRoutes": "false",
    }
    return jupiter_request("/v6/quote", params)

def jupiter_swap(
    client: SolClient,
    wallet: Keypair,
    quote: Dict[str, Any],
) -> Optional[str]:
    """
    Execute swap via Jupiter swap-instructions endpoint.
    Returns transaction signature on success, or None on failure.
    """
    if not JUPITER_API_KEY:
        print("[swap] JUPITER_API_KEY not set; cannot execute live swaps.")
        return None

    base = JUPITER_API_BASE.rstrip("/")
    url = f"{base}/v6/swap-instructions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JUPITER_API_KEY}",
    }

    payload = {
        "quoteResponse": quote,
        "userPublicKey": str(wallet.public_key),
        "wrapAndUnwrapSol": True,
        "useSharedAccounts": True,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[swap] swap-instructions error: {exc!r}")
        return None

    try:
        swap_tx_b64 = data.get("swapTransaction")
        if not swap_tx_b64:
            print("[swap] No swapTransaction returned from Jupiter.")
            return None

        tx_bytes = base64.b64decode(swap_tx_b64)
        tx = Transaction.deserialize(tx_bytes)
        tx.sign(wallet)
        raw = tx.serialize()

        sig = client.send_raw_transaction(raw)
        print(f"[swap] sent tx: {sig}")
        client.confirm_transaction(sig)
        return sig
    except Exception as exc:
        print(f"[swap] error submitting tx: {exc!r}")
        return None

# ---------- simple signal logic ----------
def pick_random_meme_token() -> Dict[str, str]:
    if not MEME_TOKENS:
        raise RuntimeError("MEME_TOKENS list is empty; add some mint addresses.")
    return random.choice(MEME_TOKENS)

def generate_demo_signal() -> Dict[str, Any]:
    """
    Placeholder for your real model / GMGN logic.
    For now: returns a fixed strong probability and random meme token.
    """
    token = pick_random_meme_token()
    prob = 0.80  # aggressive
    risk_ratio = 0.10 + random.random() * 0.20  # 10–30% of available SOL
    return {
        "token": token,
        "prob": prob,
        "risk_ratio": risk_ratio,
    }

# ---------- core trading loop ----------
def get_simulated_balance(sol_balance: float) -> float:
    return sol_balance

def compute_trade_size_sol(current_sol: float, risk_ratio: float) -> float:
    # Cap by MAX_TRADE_RISK_SOL and what's available
    target = current_sol * risk_ratio
    target = min(target, MAX_TRADE_RISK_SOL, current_sol * 0.95)
    return max(target, 0.0)

def run_once_live(wallet: Keypair, client: SolClient) -> None:
    if daily_loss_exceeded():
        return

    # Get wallet SOL balance
    try:
        balance_resp = client.get_balance(wallet.public_key)
        lamports = balance_resp["result"]["value"]
        sol_balance = lamports / 1_000_000_000
    except Exception as exc:
        print(f"[live] Error fetching balance: {exc!r}")
        return

    signal = generate_demo_signal()
    token = signal["token"]
    prob = signal["prob"]
    risk_ratio = signal["risk_ratio"]

    print(
        f"[signal] DEMO symbol={token['symbol']} prob={prob:.3f} "
        f"threshold={MIN_PROBABILITY:.3f} risk_ratio={risk_ratio:.3f}"
    )

    if prob < MIN_PROBABILITY:
        return

    trade_sol = compute_trade_size_sol(sol_balance, risk_ratio)
    if trade_sol <= 0:
        print("[live] Trade size <= 0; skipping.")
        return

    amount_lamports = int(trade_sol * 1_000_000_000)
    print(
        f"[live] Attempting trade size={trade_sol:.4f} SOL "
        f"(wallet {sol_balance:.4f} SOL) token={token['symbol']}..."
    )

    quote = get_jupiter_quote(SOL_MINT, token["mint"], amount_lamports)
    if not quote:
        print("[live] No quote from Jupiter.")
        return

    out_amount = quote.get("outAmount")
    print(f"[live] Jupiter quote OK; outAmount={out_amount}")

    sig = jupiter_swap(client, wallet, quote)
    if sig:
        print(f"[live] swap tx signature: {sig}")
    else:
        print("[live] swap failed.")

def run_once_sim(sim_state: Dict[str, Any]) -> None:
    if daily_loss_exceeded():
        return

    signal = generate_demo_signal()
    token = signal["token"]
    prob = signal["prob"]
    risk_ratio = signal["risk_ratio"]

    print(
        f"[signal] DEMO symbol={token['symbol']} prob={prob:.3f} "
        f"threshold={MIN_PROBABILITY:.3f} risk_ratio={risk_ratio:.3f}"
    )

    if prob < MIN_PROBABILITY:
        return

    sol_balance = sim_state["sol"]
    trade_sol = compute_trade_size_sol(sol_balance, risk_ratio)
    if trade_sol <= 0:
        print("[SIM] Trade size <= 0; skipping.")
        return

    amount_lamports = int(trade_sol * 1_000_000_000)
    quote = get_jupiter_quote(SOL_MINT, token["mint"], amount_lamports)
    if not quote:
        print("[SIM] No quote from Jupiter.")
        return

    # Very rough sim: assume perfect fill and +10% edge
    pnl_sol = trade_sol * 0.10
    sim_state["sol"] += pnl_sol
    new_sol = sim_state["sol"]

    pnl_usd = pnl_sol * SOL_PRICE_USD
    realized = update_realized_pnl(pnl_usd)

    print(
        f"[SIM] traded {trade_sol:.4f} SOL -> pnl={pnl_sol:.4f} "
        f"new={new_sol:.4f} | realized_today={realized:.2f} USD"
    )

# ---------- main orchestration ----------
def run_bot():
    print(
        f"[start] Memebot | SIM={SIMULATION_MODE} | "
        f"START={STARTING_SOL:.4f} SOL | MODE={START_MODE}"
    )

    if not SOLANA_OK:
        print("[solana] Solana libs not available (import failed).")
        if not SIMULATION_MODE:
            print(
                "[fatal] SIMULATION_MODE=False but SOLANA_OK=False; "
                "cannot go live until solana package is installed correctly."
            )
            raise SystemExit(1)

    if START_MODE == "withdraw":
        print("[mode] withdraw mode is not yet implemented in this simplified bot.")
        raise SystemExit(0)

    wallet = None
    client = None

    if not SIMULATION_MODE:
        wallet, client = init_solana_wallet()
        if not wallet or not client:
            print("[fatal] Wallet/client init failed; cannot go live.")
            raise SystemExit(1)

    sim_state = {"sol": STARTING_SOL}

    # Schedule loop
    def job():
        if SIMULATION_MODE:
            run_once_sim(sim_state)
        else:
            run_once_live(wallet, client)

    print(f"[job] Starting loop; SCAN_INTERVAL_SECONDS={SCAN_INTERVAL_SECONDS}")
    job()  # run immediately once at start
    schedule.every(SCAN_INTERVAL_SECONDS).seconds.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    run_bot()
