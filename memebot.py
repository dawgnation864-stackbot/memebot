"""
MemeBot – aggressive Solana memecoin bot for Railway + Jupiter Ultra

====================  ENV VARIABLES (Railway → Variables)  ====================

# run modes
SIMULATION_MODE        -> "True" or "False" (string)
START_MODE             -> "start"  (or "withdraw" – not implemented yet)

# wallet / RPC
WALLET_PRIVATE_KEY     -> base58-encoded secret key (NOT your seed phrase)
SOLANA_RPC             -> https://api.mainnet-beta.solana.com
WITHDRAWAL_ADDRESS     -> your Solana address for emergency withdraws

# risk / sizing
STARTING_SOL           -> e.g. 0.4000
DAILY_LOSS_LIMIT_USD   -> e.g. 25
SOL_PRICE_USD          -> e.g. 180  (rough PnL calc)
MAX_TRADE_RISK_SOL     -> e.g. 0.40
TAKE_PROFIT_MULT       -> e.g. 2.0  (unused; reserved)
STOP_LOSS_MULT         -> e.g. 0.5  (unused; reserved)

# signal filters
MIN_PROBABILITY        -> e.g. 0.70
SCAN_INTERVAL_SECONDS  -> e.g. 60

# Jupiter Ultra
JUPITER_API_KEY        -> your Ultra API key (from Jupiter dashboard)
JUPITER_BASE           -> https://api.jup.ag/ultra   (recommended default)

# safety / misc
PIN                    -> any 4–6 digit number you set (future use)
NEGATIVE_KEYWORDS      -> comma list, e.g. "honeypot,scam,rugpull"
MEME_TOKENS            -> optional "NAME:mint,NAME2:mint2,..."
                          if empty, DEFAULT_MEME_TOKENS is used below.

===============================================================================
"""

from __future__ import annotations

# ---------- standard library imports ----------
import os
import time
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional

# ---------- third-party imports ----------
import requests
import numpy as np
from dotenv import load_dotenv
import schedule

# ---------- optional Solana / Jupiter support ----------
SOLANA_OK = False
try:
    from solana.rpc.api import Client as RpcClient
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from base58 import b58decode

    SOLANA_OK = True
except Exception as exc:
    print(f"[solana] Import error: {exc!r}")
    SOLANA_OK = False

# ---------- load .env (for local dev; Railway uses real env) ----------
load_dotenv()

# ---------- helpers to read env ----------

def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default

# ---------- configuration from env ----------

SIMULATION_MODE = env_bool("SIMULATION_MODE", True)
START_MODE = os.getenv("START_MODE", "start").strip().lower()

SOLANA_RPC = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com").strip()
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "").strip()
WITHDRAWAL_ADDRESS = os.getenv("WITHDRAWAL_ADDRESS", "").strip()

STARTING_SOL = env_float("STARTING_SOL", 0.40)
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 180.0)
DAILY_LOSS_LIMIT_USD = env_float("DAILY_LOSS_LIMIT_USD", 25.0)
MAX_TRADE_RISK_SOL = env_float("MAX_TRADE_RISK_SOL", 0.40)
TAKE_PROFIT_MULT = env_float("TAKE_PROFIT_MULT", 2.0)
STOP_LOSS_MULT = env_float("STOP_LOSS_MULT", 0.5)

MIN_PROBABILITY = env_float("MIN_PROBABILITY", 0.70)
SCAN_INTERVAL_SECONDS = env_int("SCAN_INTERVAL_SECONDS", 60)

# Jupiter Ultra config
JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "").strip()
JUPITER_BASE = os.getenv("JUPITER_BASE", "https://api.jup.ag/ultra").strip().rstrip("/")

PIN_CODE = os.getenv("PIN", "0000").strip()
NEGATIVE_KEYWORDS = [
    k.strip().lower() for k in os.getenv("NEGATIVE_KEYWORDS", "").split(",") if k.strip()
]

# ---------- aggressive default meme list (examples only) ----------

DEFAULT_MEME_TOKENS: Dict[str, str] = {
    "BONK": "DezXAFuB81om4uPCecv9hVb2tSCD5qLQJd4d8zF9CqY",
    "WIF": "8bF4uoN9kUQJeVX5TR1fURCa8yE1xHd2h9kPGfiVNN7E",
    "POPCAT": "7aKxL5D2UzmkFbnxgSJ8tkWPcShM3SPWwHgvxCrkvV2n",
    "MEW": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "MYRO": "2RSuB8m67xY7qsKC3gQeHTFfaR9EMgrFHafpXp1em2aH",
    "BOME": "3gqVdsn9D1Gn28AL5soMgqd7qV3CyMfCVxYjByBPjVAk",
    "JEETS": "2JTSi9b3n9ee2YdzPjhzPa1L1kUD9z3iKnB2jPSj7uw6",
    "SLERF": "A5FK5GRnmt1vGjNFH6G6Dq3uTJS3BM4tiTnC5NzJctqv",
    "PENG": "9PENGQk3R3ZkN93Bp96rKMgVx7QiZfq8np5xpoE2mR7S",
    "SAMO": "7xKXtg2s9mLMpTq2s93iDby5SLmtAoeJbY7aHedj5Lwa",
    "CHONK": "4CHoNkWzHe9aVq4p6ZtUZmW8Py46nCTaPPnnw6G7xJtP",
    "TURBO": "2TuRBoS4Gf1HeLenp8QxR3cWsvtT2mG5DeADP4bVpFXZ",
    "FLOKI": "3k5Flokie3RX2iYp7Gx5oBJNrCaLCSv6iQFZjHzKzJk6",
    "MOODENG": "8Mo0DenGkGSzaRN3qHYpVXyPDbL3U5eScVn2RZLXg726",
}

MEME_TOKENS_ENV = os.getenv("MEME_TOKENS", "").strip()
if MEME_TOKENS_ENV:
    MEME_TOKENS: Dict[str, str] = {}
    for raw in MEME_TOKENS_ENV.split(","):
        part = raw.strip()
        if not part:
            continue
        if ":" in part:
            name, mint = [p.strip() for p in part.split(":", 1)]
            MEME_TOKENS[name or mint] = mint
        else:
            MEME_TOKENS[part] = part
else:
    MEME_TOKENS = DEFAULT_MEME_TOKENS

# ---------- runtime state ----------

balance_sol: float = STARTING_SOL
start_day = datetime.utcnow().date()
realized_pnl_usd: float = 0.0

# ---------- Solana helpers ----------

def init_solana_wallet():
    """Load wallet from WALLET_PRIVATE_KEY if possible."""
    if not SOLANA_OK:
        print("[wallet] SOLANA_OK=False; cannot do live trades.")
        return None, None

    if not WALLET_PRIVATE_KEY:
        print("[wallet] WALLET_PRIVATE_KEY not set; cannot go live.")
        return None, None

    try:
        secret_key_bytes = b58decode(WALLET_PRIVATE_KEY)
        wallet = Keypair.from_bytes(secret_key_bytes)
        client = RpcClient(SOLANA_RPC)
        print(f"[wallet] Loaded wallet: {wallet.pubkey()}")
        return wallet, client
    except Exception as exc:
        print(f"[wallet] Error loading wallet: {exc!r}")
        return None, None

# ---------- Jupiter helpers (Ultra v1) ----------

def jupiter_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if JUPITER_API_KEY:
        # Ultra typically uses x-api-key style
        headers["x-api-key"] = JUPITER_API_KEY
    return headers

def jupiter_quote(input_mint: str, output_mint: str, amount_lamports: int) -> Optional[Dict[str, Any]]:
    """
    Call Jupiter Ultra quote endpoint.

    NOTE: We assume JUPITER_BASE is something like:
      - https://api.jup.ag/ultra
    and we append '/v1/quote'.
    """
    base = JUPITER_BASE.rstrip("/")
    url = f"{base}/v1/quote"

    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount_lamports),
        "slippageBps": "500",
        "onlyDirectRoutes": "false",
    }

    try:
        resp = requests.get(url, params=params, headers=jupiter_headers(), timeout=15)
        if resp.status_code == 401:
            print("[swap] quote error: 401 Unauthorized from Jupiter. "
                  "Check JUPITER_API_KEY and JUPITER_BASE in Railway.")
            return None
        if resp.status_code == 404:
            print(f"[swap] quote error: 404 Not Found at {url}. "
                  "Check if Ultra v1 path is correct for your account.")
            return None
        resp.raise_for_status()
        data = resp.json()
        if not data:
            print("[swap] quote error: empty response from Jupiter.")
            return None
        return data
    except requests.exceptions.RequestException as exc:
        print(f"[swap] quote network/error: {exc!r}")
        return None
    except Exception as exc:
        print(f"[swap] quote unexpected error: {exc!r}")
        return None

def jupiter_swap(wallet: Keypair, client: RpcClient, quote: Dict[str, Any]) -> Optional[str]:
    """
    Call Jupiter Ultra swap endpoint with quote result.
    """
    base = JUPITER_BASE.rstrip("/")
    url = f"{base}/v1/swap"

    try:
        payload = {
            "quoteResponse": quote,
            "userPublicKey": str(wallet.pubkey()),
            "wrapAndUnwrapSol": True,
        }

        resp = requests.post(url, json=payload, headers=jupiter_headers(), timeout=30)
        if resp.status_code == 401:
            print("[swap] swap error: 401 Unauthorized from Jupiter. "
                  "Check JUPITER_API_KEY and JUPITER_BASE.")
            return None
        if resp.status_code == 404:
            print(f"[swap] swap error: 404 Not Found at {url}.")
            return None

        resp.raise_for_status()
        data = resp.json()
        stx = data.get("swapTransaction")
        if not stx:
            print("[swap] swapTransaction missing in Jupiter response.")
            return None

        raw_tx = base64.b64decode(stx)
        send_resp = client.send_raw_transaction(raw_tx)
        sig = send_resp.get("result") or send_resp
        print(f"[swap] submitted tx: {sig}")
        return str(sig)
    except requests.exceptions.RequestException as exc:
        print(f"[swap] swap network/error: {exc!r}")
        return None
    except Exception as exc:
        print(f"[swap] swap unexpected error: {exc!r}")
        return None

# ---------- signal + risk logic ----------

def sample_signal() -> Optional[Dict[str, Any]]:
    """
    Simple aggressive signal generator:
      - pick random meme from MEME_TOKENS
      - sample probability ~ Beta(8,2) (skewed high)
      - discard if prob < MIN_PROBABILITY
    """
    if not MEME_TOKENS:
        return None

    names: List[str] = list(MEME_TOKENS.keys())
    idx = np.random.randint(0, len(names))
    name = names[idx]
    mint = MEME_TOKENS[name]

    prob = float(np.random.beta(8, 2))
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
        print(f"[risk] daily loss cap hit "
              f"{realized_pnl_usd:.2f} USD; no more trades today.")
        return True
    return False

def trade_once():
    """
    One trading cycle: get signal → risk checks → sim or live trade.
    """
    global balance_sol, realized_pnl_usd

    print("[job] Fetching signals…")
    if should_stop_for_day():
        return

    signal = sample_signal()
    if not signal:
        print("[signal] No qualifying signal this cycle.")
        return

    name = signal["name"]
    mint = signal["mint"]
    prob = signal["prob"]
    risk_ratio = signal["risk_ratio"]

    print(f"[signal] name={name} prob={prob:.3f} risk_ratio={risk_ratio:.2f}")

    # size calculation
    trade_size_sol = min(balance_sol * risk_ratio, MAX_TRADE_RISK_SOL)
    if trade_size_sol <= 0:
        print("[trade] No SOL available to risk.")
        return

    trade_size_lamports = int(trade_size_sol * 1_000_000_000)

    # ============ SIMULATION ONLY ============
    if SIMULATION_MODE or not SOLANA_OK:
        pnl_mult = float(np.random.normal(loc=1.15, scale=0.5))
        new_balance = balance_sol - trade_size_sol + trade_size_sol * pnl_mult
        pnl_sol = new_balance - balance_sol
        balance_sol = new_balance

        pnl_usd = pnl_sol * SOL_PRICE_USD
        realized_pnl_usd += pnl_usd

        print(f"[SIM] traded {trade_size_sol:.4f} SOL → pnl={pnl_sol:.4f} SOL "
              f"new_balance={balance_sol:.4f} SOL (pnl_usd={pnl_usd:.2f})")
        return

    # ============ LIVE TRADING PATH ============

    wallet, client = init_solana_wallet()
    if not wallet or not client:
        print("[live] Wallet/client unavailable; skipping trade.")
        return

    if not JUPITER_API_KEY:
        print("[live] JUPITER_API_KEY not set; cannot call Ultra endpoints.")
        return

    SOL_MINT = "So11111111111111111111111111111111111111112"

    quote = jupiter_quote(SOL_MINT, mint, trade_size_lamports)
    if not quote:
        print("[live] No viable route from Jupiter (quote failed).")
        return

    sig = jupiter_swap(wallet, client, quote)
    if not sig:
        print("[live] Swap failed; no funds moved (as far as we know).")
        return

    print(f"[LIVE] swap submitted for {trade_size_sol:.4f} SOL "
          f"into {name} mint={mint}, tx={sig}")

# ---------- main loop ----------

def main():
    print(">>> memebot.py starting")
    print(
        f"[start] SIMULATION_MODE={SIMULATION_MODE} "
        f"STARTING_SOL={STARTING_SOL:.4f}"
    )

    if not SOLANA_OK and not SIMULATION_MODE:
        print("[fatal] SIMULATION_MODE=False but Solana libs are missing. "
              "Either set SIMULATION_MODE=True or add solana + solders to requirements.")
        return

    if START_MODE == "withdraw":
        print("[withdraw] START_MODE=withdraw not implemented in this version.")
        return

    # schedule job
    schedule.every(SCAN_INTERVAL_SECONDS).seconds.do(trade_once)

    # run once immediately
    trade_once()

    # loop forever
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
