"""
MemeBot – aggressive Solana memecoin bot for Railway + Jupiter

REQUIRED ENV VARS (Railway → Variables)
---------------------------------------

# run modes
SIMULATION_MODE        -> "True" or "False"  (string)
START_MODE             -> "start"  (or "withdraw" – not implemented yet)

# wallet / RPC
WALLET_PRIVATE_KEY     -> base58-encoded secret key (NOT your seed phrase)
SOLANA_RPC             -> https://api.mainnet-beta.solana.com
WITHDRAWAL_ADDRESS     -> your Solana address for emergency withdraws

# risk / sizing
STARTING_SOL           -> e.g. 0.4000
DAILY_LOSS_LIMIT_USD   -> e.g. 25
SOL_PRICE_USD          -> e.g. 180  (rough pnl calc only)
MAX_TRADE_RISK_SOL     -> e.g. 0.40
TAKE_PROFIT_MULT       -> e.g. 2.0   (2x)
STOP_LOSS_MULT         -> e.g. 0.5   (-50%)

# signal filters
MIN_PROBABILITY        -> e.g. 0.70
SCAN_INTERVAL_SECONDS  -> e.g. 60

# Jupiter API
#  Option A: public v6 (NO key needed)
#     JUPITER_API_BASE  = https://quote-api.jup.ag
#     JUPITER_QUOTE_PATH= /v6/quote
#     JUPITER_SWAP_PATH = /v6/swap
#     JUPITER_API_KEY   =  (empty)
#
#  Option B: Ultra v1 (requires key + plan)
#     JUPITER_API_BASE  = https://api.jup.ag/ultra/v1
#     JUPITER_QUOTE_PATH= /quote
#     JUPITER_SWAP_PATH = /swap
#     JUPITER_API_KEY   = your Ultra key
#
JUPITER_API_BASE       -> see above
JUPITER_QUOTE_PATH     -> see above
JUPITER_SWAP_PATH      -> see above
JUPITER_API_KEY        -> your key (or blank for public)

# safety / misc
PIN                    -> any 4–6 digit number you set (for future controls)
NEGATIVE_KEYWORDS      -> comma list, e.g. "honeypot,scam,rugpull"
MEME_TOKENS            -> optional comma list of mint addresses
                         formats:
                           "BONK:mint,WIF:mint,..." or
                           "mint1,mint2,..."
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

# ---------- optional Solana / Jupiter libs ----------
SOLANA_OK = False
try:
    from solana.rpc.api import Client as RpcClient
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey

    SOLANA_OK = True
except Exception as exc:
    print(f"[solana] Import error: {exc!r}")
    SOLANA_OK = False

# ---------- env helpers ----------
load_dotenv()


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


# ---------- config from env ----------

SIMULATION_MODE = env_bool("SIMULATION_MODE", True)
START_MODE = os.getenv("START_MODE", "start").strip().lower()

SOLANA_RPC = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com").strip()
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "").strip()
WITHDRAWAL_ADDRESS = os.getenv("WITHDRAWAL_ADDRESS", "").strip()

STARTING_SOL = env_float("STARTING_SOL", 0.4)
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 180.0)
DAILY_LOSS_LIMIT_USD = env_float("DAILY_LOSS_LIMIT_USD", 25.0)
MAX_TRADE_RISK_SOL = env_float("MAX_TRADE_RISK_SOL", 0.40)
TAKE_PROFIT_MULT = env_float("TAKE_PROFIT_MULT", 2.0)
STOP_LOSS_MULT = env_float("STOP_LOSS_MULT", 0.5)

MIN_PROBABILITY = env_float("MIN_PROBABILITY", 0.7)
SCAN_INTERVAL_SECONDS = int(env_float("SCAN_INTERVAL_SECONDS", 60))

# Jupiter configuration – **all HTTP issues are here**
JUPITER_API_BASE = os.getenv("JUPITER_API_BASE", "https://quote-api.jup.ag").rstrip("/")
JUPITER_QUOTE_PATH = os.getenv("JUPITER_QUOTE_PATH", "/v6/quote")
JUPITER_SWAP_PATH = os.getenv("JUPITER_SWAP_PATH", "/v6/swap")
JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "").strip()

PIN_CODE = os.getenv("PIN", "0000").strip()
NEGATIVE_KEYWORDS = [
    k.strip().lower()
    for k in os.getenv("NEGATIVE_KEYWORDS", "").split(",")
    if k.strip()
]

# Aggressive default meme list – EXAMPLE mints, not investment advice
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
balance_sol = STARTING_SOL
start_day = datetime.utcnow().date()
realized_pnl_usd = 0.0


# ---------- Solana helpers ----------

def init_solana_wallet():
    """Load wallet from WALLET_PRIVATE_KEY if libs and key are available."""
    if not SOLANA_OK:
        print("[wallet] SOLANA_OK=False, cannot init wallet (libs missing).")
        return None, None

    if not WALLET_PRIVATE_KEY:
        print("[wallet] WALLET_PRIVATE_KEY not set; cannot go live.")
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


# ---------- Jupiter HTTP helpers ----------

def build_jupiter_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if JUPITER_API_KEY:
        # Jupiter generally expects this header for API keys
        headers["x-api-key"] = JUPITER_API_KEY
    return headers


def jupiter_quote(input_mint: str, output_mint: str, amount_lamports: int) -> Optional[Dict[str, Any]]:
    """
    Ask Jupiter for best route.
    Returns a route dict compatible with /swap, or None if nothing / HTTP error.
    """
    base = JUPITER_API_BASE.rstrip("/")
    path = JUPITER_QUOTE_PATH if JUPITER_QUOTE_PATH.startswith("/") else "/" + JUPITER_QUOTE_PATH
    url = f"{base}{path}"

    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount_lamports),
        "slippageBps": "500",
        "onlyDirectRoutes": "false",
    }

    try:
        resp = requests.get(url, params=params, headers=build_jupiter_headers(), timeout=20)

        if resp.status_code == 401:
            print("[swap] Jupiter 401 Unauthorized – this is NOT a Python error.")
            print("       → Check JUPITER_API_KEY, plan status, and allowed endpoint in your Jupiter dashboard.")
            return None

        if resp.status_code == 404:
            print(f"[swap] Jupiter 404 Not Found for {url} – most likely wrong QUOTE path/base.")
            print("       → Verify JUPITER_API_BASE + JUPITER_QUOTE_PATH match the docs.")
            return None

        if resp.status_code >= 500:
            print(f"[swap] Jupiter server error {resp.status_code}: will skip this trade.")
            return None

        resp.raise_for_status()
        data = resp.json()

        # Public v6:  {"data": [ route, ... ]}
        # Ultra v1 :  may already be a route dict
        if isinstance(data, dict) and "data" in data:
            routes = data.get("data") or []
            if not routes:
                print("[swap] Quote OK but no routes returned.")
                return None
            return routes[0]

        if not data:
            print("[swap] Empty quote response.")
            return None

        return data

    except requests.exceptions.RequestException as exc:
        print(f"[swap] quote network/error: {exc!r}")
        return None


def jupiter_swap(wallet: Keypair, client: RpcClient, route: Dict[str, Any]) -> Optional[str]:
    """
    Submit a Jupiter swap using the configured SWAP endpoint.
    If anything fails, we log and return None (assume no funds moved).
    """
    base = JUPITER_API_BASE.rstrip("/")
    path = JUPITER_SWAP_PATH if JUPITER_SWAP_PATH.startswith("/") else "/" + JUPITER_SWAP_PATH
    url = f"{base}{path}"

    try:
        user_pubkey = str(wallet.pubkey())
        payload = {
            "quoteResponse": route,
            "userPublicKey": user_pubkey,
            "wrapAndUnwrapSol": True,
        }

        resp = requests.post(url, json=payload, headers=build_jupiter_headers(), timeout=30)

        if resp.status_code == 401:
            print("[swap] Jupiter 401 Unauthorized on /swap – check your key/plan.")
            return None

        if resp.status_code == 404:
            print(f"[swap] Jupiter 404 Not Found for {url} – wrong SWAP path/base.")
            return None

        if resp.status_code >= 500:
            print(f"[swap] Jupiter swap server error {resp.status_code}: {resp.text[:200]}")
            return None

        resp.raise_for_status()
        data = resp.json()

        if "swapTransaction" not in data:
            print("[swap] swapTransaction missing from response.")
            return None

        swap_tx = data["swapTransaction"]
        raw_tx = base64.b64decode(swap_tx)

        send_resp = client.send_raw_transaction(raw_tx)
        sig = send_resp.get("result")
        print(f"[swap] submitted tx: {sig}")
        return sig

    except Exception as exc:
        print(f"[swap] swap error: {exc!r}")
        return None


# ---------- signal + risk logic ----------

def sample_signal() -> Optional[Dict[str, Any]]:
    """
    Very simple 'aggressive' signal generator:
    - picks a random meme from list
    - assigns probability from 0.0 – 1.0
    - if above MIN_PROBABILITY, returns a 'buy' signal
    """
    if not MEME_TOKENS:
        return None

    names = list(MEME_TOKENS.keys())
    idx = np.random.randint(0, len(names))
    name = names[idx]
    mint = MEME_TOKENS[name]

    prob = float(np.random.beta(8, 2))  # skewed high → aggressive
    risk_ratio = float(np.random.uniform(0.3, 1.0))

    if prob < MIN_PROBABILITY:
        return None

    return {"name": name, "mint": mint, "prob": prob, "risk_ratio": risk_ratio}


def should_stop_for_day() -> bool:
    """Check daily loss cap."""
    global start_day, realized_pnl_usd
    today = datetime.utcnow().date()
    if today != start_day:
        start_day = today
        realized_pnl_usd = 0.0
        return False

    if realized_pnl_usd <= -DAILY_LOSS_LIMIT_USD:
        print(f"[risk] Daily loss cap hit {realized_pnl_usd:.2f} USD; no more trades today.")
        return True
    return False


def trade_once():
    """One trading cycle: get signal → risk checks → (live or sim) trade."""
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

    # position size in SOL
    trade_size_sol = min(balance_sol * risk_ratio, MAX_TRADE_RISK_SOL)
    if trade_size_sol <= 0:
        print("[trade] No SOL available to risk.")
        return

    trade_size_lamports = int(trade_size_sol * 1_000_000_000)

    if SIMULATION_MODE or not SOLANA_OK:
        # simulated PnL
        pnl_mult = float(np.random.normal(loc=1.15, scale=0.5))
        new_balance = balance_sol - trade_size_sol + trade_size_sol * pnl_mult
        pnl_sol = new_balance - balance_sol
        balance_sol = new_balance

        pnl_usd = pnl_sol * SOL_PRICE_USD
        realized_pnl_usd += pnl_usd

        print(
            f"[SIM] traded {trade_size_sol:.4f} → pnl={pnl_sol:.4f} "
            f"new={balance_sol:.4f} SOL (pnl_usd={pnl_usd:.2f})"
        )
        return

    # ------- LIVE TRADING PATH -------

    wallet, client = init_solana_wallet()
    if not wallet or not client:
        print("[live] Wallet/client not available; skipping live trade.")
        return

    SOL_MINT = "So11111111111111111111111111111111111111112"
    route = jupiter_quote(SOL_MINT, mint, trade_size_lamports)
    if not route:
        print("[live] No viable route from Jupiter (quote failed).")
        return

    sig = jupiter_swap(wallet, client, route)
    if not sig:
        print("[live] Swap failed or not accepted; assume no funds moved.")
        return

    print(f"[LIVE] swap submitted for {trade_size_sol:.4f} SOL into {name} mint={mint}, tx={sig}")


# ---------- main loop ----------

def main():
    print(">>> memebot.py starting")
    print(
        f"[start] SIMULATION_MODE={SIMULATION_MODE} STARTING_SOL={STARTING_SOL:.4f} "
        f"MODE={START_MODE}"
    )
    print(
        f"[jupiter] base={JUPITER_API_BASE} quote_path={JUPITER_QUOTE_PATH} "
        f"swap_path={JUPITER_SWAP_PATH} key_present={bool(JUPITER_API_KEY)}"
    )

    if not SOLANA_OK and not SIMULATION_MODE:
        print(
            "[fatal] SIMULATION_MODE=False but solana/solders not installed; "
            "cannot go live."
        )
        return

    if START_MODE == "withdraw":
        print("[withdraw] START_MODE=withdraw not implemented yet.")
        return

    # schedule trading job
    schedule.every(SCAN_INTERVAL_SECONDS).seconds.do(trade_once)

    # run once immediately
    trade_once()

    # main loop
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
