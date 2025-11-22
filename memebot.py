"""
MemeBot – aggressive Solana memecoin bot for Railway + Jupiter v6 (stable)

REQUIRED ENV VARIABLES (Railway → Variables):

# core run modes
SIMULATION_MODE        -> "True" or "False"
START_MODE             -> "start"   (or "withdraw" for future support)

# wallet / RPC
WALLET_PRIVATE_KEY     -> base58-encoded secret key (NOT your seed phrase)
SOLANA_RPC             -> https://api.mainnet-beta.solana.com
WITHDRAWAL_ADDRESS     -> your Solana address for emergency withdraws

# risk / sizing
STARTING_SOL           -> e.g. 0.4000
DAILY_LOSS_LIMIT_USD   -> e.g. 25
SOL_PRICE_USD          -> e.g. 180  (rough PnL calc)
MAX_TRADE_RISK_SOL     -> e.g. 0.40
TAKE_PROFIT_MULT       -> e.g. 2.0   (2x)
STOP_LOSS_MULT         -> e.g. 0.5   (-50%)

# signal filters
MIN_PROBABILITY        -> e.g. 0.70
SCAN_INTERVAL_SECONDS  -> e.g. 60

# Jupiter (v6 stable)
JUPITER_API_KEY        -> optional; mostly used for Ultra, harmless here
JUPITER_QUOTE_URL      -> optional; default: https://quote-api.jup.ag/v6/quote
JUPITER_SWAP_URL       -> optional; default: https://api.jup.ag/v6/swap

# safety / misc
PIN                    -> any 4–6 digit number you set (for future controls)
NEGATIVE_KEYWORDS      -> comma list, e.g. "honeypot,scam,rugpull"
MEME_TOKENS            -> optional comma list of "NAME:mint" or just "mint";
                          if empty, defaults below are used.
"""

from __future__ import annotations

# ---------- standard library imports ----------
import os
import asyncio
import base64
import time
from datetime import datetime, date
from typing import Dict, Any, List, Optional

# ---------- third-party imports ----------
import requests
import numpy as np
from dotenv import load_dotenv
import schedule

# ---------- optional Solana / solders support ----------
SOLANA_OK = False
try:
    from solana.rpc.api import Client as RpcClient
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey

    SOLANA_OK = True
except Exception as exc:  # noqa: BLE001
    print(f"[solana] Import error: {exc!r}")
    SOLANA_OK = False

# ---------- load .env (local dev) ----------
load_dotenv()

# ---------- helpers for env parsing ----------

def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}

def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    try:
        return float(raw) if raw.strip() != "" else default
    except Exception:  # noqa: BLE001
        return default


# ---------- configuration from env ----------

SIMULATION_MODE: bool = env_bool("SIMULATION_MODE", True)
START_MODE: str = os.getenv("START_MODE", "start").strip().lower()

SOLANA_RPC: str = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com").strip()
WALLET_PRIVATE_KEY: str = os.getenv("WALLET_PRIVATE_KEY", "").strip()
WITHDRAWAL_ADDRESS: str = os.getenv("WITHDRAWAL_ADDRESS", "").strip()

STARTING_SOL: float = env_float("STARTING_SOL", 0.4)
SOL_PRICE_USD: float = env_float("SOL_PRICE_USD", 180.0)
DAILY_LOSS_LIMIT_USD: float = env_float("DAILY_LOSS_LIMIT_USD", 25.0)
MAX_TRADE_RISK_SOL: float = env_float("MAX_TRADE_RISK_SOL", 0.40)
TAKE_PROFIT_MULT: float = env_float("TAKE_PROFIT_MULT", 2.0)
STOP_LOSS_MULT: float = env_float("STOP_LOSS_MULT", 0.5)

MIN_PROBABILITY: float = env_float("MIN_PROBABILITY", 0.7)
SCAN_INTERVAL_SECONDS: int = int(env_float("SCAN_INTERVAL_SECONDS", 60))

# Jupiter v6 stable endpoints
JUPITER_API_KEY: str = os.getenv("JUPITER_API_KEY", "").strip()
JUPITER_QUOTE_URL: str = os.getenv("JUPITER_QUOTE_URL", "https://quote-api.jup.ag/v6/quote").strip()
JUPITER_SWAP_URL: str = os.getenv("JUPITER_SWAP_URL", "https://api.jup.ag/v6/swap").strip()

PIN_CODE: str = os.getenv("PIN", "0000").strip()
NEGATIVE_KEYWORDS: List[str] = [
    k.strip().lower()
    for k in os.getenv("NEGATIVE_KEYWORDS", "").split(",")
    if k.strip()
]

# ---------- aggressive default meme list ----------
# NOTE: these are EXAMPLE mint addresses; replace with tokens you actually want.
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

# ---------- bot state ----------
balance_sol: float = STARTING_SOL
start_day: date = datetime.utcnow().date()
realized_pnl_usd: float = 0.0

# SOL mint constant
SOL_MINT: str = "So11111111111111111111111111111111111111112"


# =====================================================================
#                       SOLANA / JUPITER HELPERS
# =====================================================================

def init_solana_wallet() -> tuple[Optional[Keypair], Optional[RpcClient]]:
    """Load wallet from WALLET_PRIVATE_KEY if libs and key are available."""
    if not SOLANA_OK:
        print("[wallet] SOLANA_OK=False, cannot init wallet.")
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
    except Exception as exc:  # noqa: BLE001
        print(f"[wallet] Error loading wallet: {exc!r}")
        return None, None


def jupiter_quote(input_mint: str, output_mint: str, amount_lamports: int) -> Optional[Dict[str, Any]]:
    """
    Call Jupiter v6 quote endpoint.
    Returns the first route object, or None on failure.
    """
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount_lamports),
        "slippageBps": "500",          # 5% slippage
        "onlyDirectRoutes": "false",   # allow routed paths
    }

    headers = {
        "Accept": "application/json",
    }
    if JUPITER_API_KEY:
        # For v6 this is usually ignored, but harmless:
        headers["Authorization"] = f"Bearer {JUPITER_API_KEY}"

    try:
        resp = requests.get(JUPITER_QUOTE_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        print(f"[swap] quote error: HTTPError({exc.response.status_code}) for url: {resp.url}")
        try:
            print(f"[swap] body: {resp.text}")
        except Exception:  # noqa: BLE001
            pass
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[swap] quote error: {exc!r}")
        return None

    if not isinstance(data, dict):
        print("[swap] quote error: unexpected JSON type.")
        return None

    route = data.get("data") or data.get("route") or data.get("routePlan")
    # Jupiter v6 returns {"data":[{...}, {...}]}
    if isinstance(route, list) and route:
        return route[0]
    if isinstance(route, dict):
        return route

    print("[swap] quote error: no route in response.")
    return None


def jupiter_swap(wallet: Keypair, client: RpcClient, route: Dict[str, Any]) -> Optional[str]:
    """
    Submit a Jupiter swap using v6 /swap.
    If anything fails, log and return None.
    """
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if JUPITER_API_KEY:
        headers["Authorization"] = f"Bearer {JUPITER_API_KEY}"

    payload = {
        "quoteResponse": route,
        "userPublicKey": str(wallet.pubkey()),
        "wrapAndUnwrapSol": True,
    }

    try:
        resp = requests.post(JUPITER_SWAP_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        print(f"[swap] swap error: HTTPError({exc.response.status_code}) for url: {resp.url}")
        try:
            print(f"[swap] body: {resp.text}")
        except Exception:  # noqa: BLE001
            pass
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[swap] swap error: {exc!r}")
        return None

    swap_tx_b64 = data.get("swapTransaction")
    if not swap_tx_b64:
        print("[swap] swapTransaction missing from response.")
        return None

    try:
        raw_tx = base64.b64decode(swap_tx_b64)
        send_resp = client.send_raw_transaction(raw_tx)
        sig = send_resp.get("result") or send_resp
        print(f"[swap] submitted tx: {sig}")
        return str(sig)
    except Exception as exc:  # noqa: BLE001
        print(f"[swap] error sending transaction: {exc!r}")
        return None


# =====================================================================
#                        SIGNAL + TRADING LOGIC
# =====================================================================

def sample_signal() -> Optional[Dict[str, Any]]:
    """
    Very simple 'aggressive' signal generator:
    - picks a random meme from list
    - assigns probability from 0.0 – 1.0 (beta(8,2) skewed high)
    - if above MIN_PROBABILITY, returns a 'buy' signal dict
    """
    if not MEME_TOKENS:
        return None

    names = list(MEME_TOKENS.keys())
    idx = np.random.randint(0, len(names))
    name = names[idx]
    mint = MEME_TOKENS[name]

    prob = float(np.random.beta(8, 2))  # aggressive bias
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
    """Check daily loss cap and reset on new UTC day."""
    global start_day, realized_pnl_usd

    today = datetime.utcnow().date()
    if today != start_day:
        start_day = today
        realized_pnl_usd = 0.0
        return False

    if realized_pnl_usd <= -DAILY_LOSS_LIMIT_USD:
        print(f"[risk] daily loss cap hit {realized_pnl_usd:.2f} USD; no more trades today.")
        return True

    return False


def trade_once() -> None:
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

    print(f"[signal] DEMO name={name} prob={prob:.3f} risk_ratio={risk_ratio:.2f}")

    # position size in SOL
    trade_size_sol = min(balance_sol * risk_ratio, MAX_TRADE_RISK_SOL)
    if trade_size_sol <= 0:
        print("[trade] No SOL available to risk.")
        return

    trade_size_lamports = int(trade_size_sol * 1_000_000_000)

    # ---------------- SIMULATION PATH ----------------
    if SIMULATION_MODE or not SOLANA_OK:
        pnl_mult = float(np.random.normal(loc=1.15, scale=0.5))  # aggressive distribution
        new_balance = balance_sol - trade_size_sol + trade_size_sol * pnl_mult
        pnl_sol = new_balance - balance_sol
        balance_sol = new_balance

        pnl_usd = pnl_sol * SOL_PRICE_USD
        realized_pnl_usd += pnl_usd

        print(
            f"[SIM] traded {trade_size_sol:.4f} SOL "
            f"→ pnl={pnl_sol:.4f} new_balance={balance_sol:.4f} SOL "
            f"(pnl_usd={pnl_usd:.2f})"
        )
        return

    # ---------------- LIVE TRADING PATH ----------------

    wallet, client = init_solana_wallet()
    if not wallet or not client:
        print("[live] Wallet/client not available; skipping live trade.")
        return

    # Get quote from Jupiter v6
    route = jupiter_quote(SOL_MINT, mint, trade_size_lamports)
    if not route:
        print("[live] No viable route from Jupiter (quote failed).")
        return

    sig = jupiter_swap(wallet, client, route)
    if not sig:
        print("[live] Swap failed; no funds moved (as far as we know).")
        return

    print(
        f"[LIVE] swap submitted for {trade_size_sol:.4f} SOL into {name} "
        f"mint={mint}, tx={sig}"
    )
    # For now we’re not updating on-chain PnL; you can add that later.


# =====================================================================
#                               MAIN
# =====================================================================

def main() -> None:
    print(">>> memebot.py starting")
    print(
        f"[start] SIMULATION_MODE={SIMULATION_MODE} "
        f"STARTING_SOL={STARTING_SOL:.4f} "
        f"JUPITER_QUOTE_URL={JUPITER_QUOTE_URL}"
    )

    if not SOLANA_OK and not SIMULATION_MODE:
        print(
            "[fatal] SIMULATION_MODE=False but SOLANA_OK=False; "
            "cannot go live until solana + solders are installed correctly."
        )
        return

    if START_MODE == "withdraw":
        print("[withdraw] START_MODE=withdraw not yet implemented in this version.")
        return

    # schedule trading job
    schedule.every(SCAN_INTERVAL_SECONDS).seconds.do(trade_once)

    # run immediately once on start
    trade_once()

    # main loop
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
