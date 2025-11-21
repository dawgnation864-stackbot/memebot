"""
MemeBot – aggressive Solana memecoin bot for Railway + Jupiter (PUBLIC API, no Ultra)

This version:
- Uses the public Jupiter aggregator (https://quote-api.jup.ag)
- Does NOT require any Jupiter API key
- Trades random meme tokens from a list, in SOL
- Respects SIMULATION_MODE (True/False) via env vars

REQUIRED ENV VARIABLES (Railway → Variables):

# core run modes
SIMULATION_MODE        -> "True" or "False"   (string)
START_MODE             -> "start"            (or "withdraw" in future)

# wallet / RPC
WALLET_PRIVATE_KEY     -> base58-encoded secret key (NOT your seed phrase)
SOLANA_RPC             -> https://api.mainnet-beta.solana.com
WITHDRAWAL_ADDRESS     -> your Solana address for emergency withdraws (not used yet in this version)

# risk / sizing
STARTING_SOL           -> e.g. 0.4000
DAILY_LOSS_LIMIT_USD   -> e.g. 25
SOL_PRICE_USD          -> e.g. 180  (rough PnL calc)
MAX_TRADE_RISK_SOL     -> e.g. 0.40

# signal filters
MIN_PROBABILITY        -> e.g. 0.70
SCAN_INTERVAL_SECONDS  -> e.g. 60   (how often to try a trade)

# Jupiter public
JUPITER_BASE           -> https://quote-api.jup.ag   (no key needed)
"""

from __future__ import annotations

# ---------- standard library ----------
import os
import time
import base64
from datetime import datetime
from typing import Dict, Any, Optional

# ---------- third-party ----------
import requests
import numpy as np
from dotenv import load_dotenv
import schedule

# ---------- Solana libs (must be in requirements and installed) ----------
SOLANA_OK = False
try:
    from solana.rpc.api import Client as RpcClient
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    SOLANA_OK = True
except Exception as exc:
    print(f"[solana] Import error: {exc!r}")
    SOLANA_OK = False

# ---------- load env ----------
load_dotenv()

# ---------- env helpers ----------
def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# ---------- config ----------
SIMULATION_MODE = env_bool("SIMULATION_MODE", True)
START_MODE = os.getenv("START_MODE", "start").strip().lower()

SOLANA_RPC = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com").strip()
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "").strip()
WITHDRAWAL_ADDRESS = os.getenv("WITHDRAWAL_ADDRESS", "").strip()

STARTING_SOL = env_float("STARTING_SOL", 0.4)
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 180.0)
DAILY_LOSS_LIMIT_USD = env_float("DAILY_LOSS_LIMIT_USD", 25.0)
MAX_TRADE_RISK_SOL = env_float("MAX_TRADE_RISK_SOL", 0.40)

MIN_PROBABILITY = env_float("MIN_PROBABILITY", 0.7)
SCAN_INTERVAL_SECONDS = int(env_float("SCAN_INTERVAL_SECONDS", 60))

# Jupiter public base (no key needed)
JUPITER_BASE = os.getenv("JUPITER_BASE", "https://quote-api.jup.ag").rstrip("/")

# ---------- meme token list ----------
DEFAULT_MEME_TOKENS: Dict[str, str] = {
    "BONK":   "DezXAFuB81om4uPCecv9hVb2tSCD5qLQJd4d8zF9CqY",
    "WIF":    "8bF4uoN9kUQJeVX5TR1FURCa8yE1xHd2h9kPGfiVNN7E",
    "POPCAT": "7aKxL5D2UzmkFbnxgSJ8tkWPcShM3SPWwHgvxCrkvV2n",
    "MEW":    "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "MYRO":   "2RSuB8m67xY7qsKC3gQeHTFfaR9EMgrFHafpXp1em2aH",
    "BOME":   "3gqVdsn9D1Gn28AL5soMgqd7qV3CyMfCVxYjByBPjVAk",
    "JEETS":  "2JTSi9b3n9ee2YdzPjhzPa1L1kUD9z3iKnB2jPSj7uw6",
    "SLERF":  "A5FK5GRnmt1vGjNFH6G6Dq3uTJS3BM4tiTnC5NzJctqv",
    "PENG":   "9PENGQk3R3ZkN93Bp96rKMgVx7QiZfq8np5xpoE2mR7S",
    "SAMO":   "7xKXtg2s9mLMpTq2s93iDby5SLmtAoeJbY7aHedj5Lwa",
}

MEME_TOKENS = DEFAULT_MEME_TOKENS

# ---------- state ----------
balance_sol: float = STARTING_SOL
start_day = datetime.utcnow().date()
realized_pnl_usd: float = 0.0

# ---------- Solana + wallet ----------
def init_solana_wallet() -> tuple[Optional[Keypair], Optional[RpcClient]]:
    if not SOLANA_OK:
        print("[wallet] Solana libs not available.")
        return None, None
    if not WALLET_PRIVATE_KEY:
        print("[wallet] WALLET_PRIVATE_KEY not set.")
        return None, None

    try:
        from base58 import b58decode
        secret_key_bytes = b58decode(WALLET_PRIVATE_KEY)
        wallet = Keypair.from_bytes(secret_key_bytes)
        client = RpcClient(SOLANA_RPC)
        print(f"[wallet] Loaded wallet: {wallet.pubkey()}")
        return wallet, client
    except Exception as exc:
        print(f"[wallet] Error loading wallet: {exc!r}")
        return None, None

# ---------- Jupiter public helpers ----------
def jupiter_quote(input_mint: str, output_mint: str, amount_lamports: int) -> Optional[Dict[str, Any]]:
    """
    Call Jupiter public v6 quote endpoint (no auth).
    Returns a dict with route info or None.
    """
    url = f"{JUPITER_BASE}/v6/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount_lamports),
        "slippageBps": 500,
        "onlyDirectRoutes": "false",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Jupiter can return { "data": [routes...] } or a single route
        if isinstance(data, dict) and "data" in data:
            routes = data.get("data") or []
            if not routes:
                return None
            return routes[0]
        # fallback if already a route
        return data
    except Exception as exc:
        print(f"[swap] quote error: {exc!r}")
        return None

def jupiter_swap(wallet: Keypair, client: RpcClient, route: Dict[str, Any]) -> Optional[str]:
    """
    Submit a Jupiter swap using /v6/swap.
    """
    try:
        url = f"{JUPITER_BASE}/v6/swap"
        user_pubkey = str(wallet.pubkey())
        payload = {
            "quoteResponse": route,
            "userPublicKey": user_pubkey,
            "wrapAndUnwrapSol": True,
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "swapTransaction" not in data:
            print("[swap] swapTransaction missing from response.")
            return None

        swap_tx = data["swapTransaction"]
        raw_tx = base64.b64decode(swap_tx)
        send_resp = client.send_raw_transaction(raw_tx)
        sig = send_resp.get("result") or send_resp
        print(f"[swap] submitted tx: {sig}")
        return sig
    except Exception as exc:
        print(f"[swap] swap error: {exc!r}")
        return None

# ---------- signals + trading ----------
def sample_signal() -> Optional[Dict[str, Any]]:
    """
    Picks a random meme and assigns a 'probability' 0–1.
    If prob >= MIN_PROBABILITY, returns a buy signal.
    """
    if not MEME_TOKENS:
        return None

    names = list(MEME_TOKENS.keys())
    idx = np.random.randint(0, len(names))
    name = names[idx]
    mint = MEME_TOKENS[name]
    prob = float(np.random.beta(8, 2))  # skewed high = aggressive
    risk_ratio = float(np.random.uniform(0.3, 1.0))

    if prob < MIN_PROBABILITY:
        return None

    return {
        "name": name,
        "mint": mint,
        "prob": prob,
        "risk_ratio": risk_ratio,
    }

def should_stop_for_day() -> bool:
    global start_day, realized_pnl_usd
    today = datetime.utcnow().date()
    if today != start_day:
        start_day = today
        realized_pnl_usd = 0.0
        return False
    if realized_pnl_usd <= -DAILY_LOSS_LIMIT_USD:
        print(f"[risk] daily loss cap hit {realized_pnl_usd:.2f} USD; stopping for today.")
        return True
    return False

def trade_once():
    """
    One cycle: get a signal, check risk, then simulate or trade live.
    """
    global balance_sol, realized_pnl_usd

    print("[job] Fetching signals…")
    if should_stop_for_day():
        return

    sig = sample_signal()
    if not sig:
        print("[signal] No qualifying signal this cycle.")
        return

    name = sig["name"]
    mint = sig["mint"]
    prob = sig["prob"]
    risk_ratio = sig["risk_ratio"]

    print(f"[signal] name={name} prob={prob:.3f} risk_ratio={risk_ratio:.2f}")

    # position size
    trade_size_sol = min(balance_sol * risk_ratio, MAX_TRADE_RISK_SOL)
    if trade_size_sol <= 0:
        print("[trade] No SOL to risk.")
        return

    trade_size_lamports = int(trade_size_sol * 1_000_000_000)

    if SIMULATION_MODE or not SOLANA_OK:
        # aggressive random PnL
        pnl_mult = float(np.random.normal(loc=1.15, scale=0.5))
        new_balance = balance_sol - trade_size_sol + trade_size_sol * pnl_mult
        pnl_sol = new_balance - balance_sol
        balance_sol = new_balance
        pnl_usd = pnl_sol * SOL_PRICE_USD
        realized_pnl_usd += pnl_usd
        print(f"[SIM] traded {trade_size_sol:.4f} → pnl={pnl_sol:.4f} new={balance_sol:.4f} SOL (~${pnl_usd:.2f})")
        return

    # LIVE PATH
    wallet, client = init_solana_wallet()
    if not wallet or not client:
        print("[live] No wallet/client; skipping live trade.")
        return

    SOL_MINT = "So11111111111111111111111111111111111111112"

    route = jupiter_quote(SOL_MINT, mint, trade_size_lamports)
    if not route:
        print("[live] No viable route from Jupiter.")
        return

    sig = jupiter_swap(wallet, client, route)
    if not sig:
        print("[live] Swap failed; no funds moved (as far as we know).")
        return

    print(f"[LIVE] swap submitted: {sig} | {trade_size_sol:.4f} SOL into {name} ({mint})")

# ---------- main ----------
def main():
    print(">>> memebot.py starting")
    print(f"[start] SIMULATION_MODE={SIMULATION_MODE} STARTING_SOL={STARTING_SOL:.4f}")

    if not SOLANA_OK and not SIMULATION_MODE:
        print("[fatal] No Solana libs and SIMULATION_MODE=False. Install packages or enable simulation.")
        return

    if START_MODE == "withdraw":
        print("[withdraw] Withdraw logic not implemented in this version.")
        return

    # schedule
    schedule.every(SCAN_INTERVAL_SECONDS).seconds.do(trade_once)

    # run once immediately
    trade_once()

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
