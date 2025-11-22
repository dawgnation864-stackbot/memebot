"""
MemeBot – aggressive Solana memecoin bot for Railway + Jupiter PUBLIC API

Set these ENV VARIABLES in Railway:

# core run modes
SIMULATION_MODE        -> "True" or "False"   (string)
START_MODE             -> "start"            (or "withdraw" – not implemented yet)

# wallet / RPC (required for live trading)
WALLET_PRIVATE_KEY     -> base58-encoded secret key (NOT your seed phrase)
SOLANA_RPC             -> https://api.mainnet-beta.solana.com
WITHDRAWAL_ADDRESS     -> your Solana address (for future withdraw feature)

# risk / sizing
STARTING_SOL           -> e.g. 0.4000
DAILY_LOSS_LIMIT_USD   -> e.g. 25
SOL_PRICE_USD          -> e.g. 180   (rough PnL calc only)
MAX_TRADE_RISK_SOL     -> e.g. 0.40
TAKE_PROFIT_MULT       -> e.g. 2.0
STOP_LOSS_MULT         -> e.g. 0.5

# signal filters
MIN_PROBABILITY        -> e.g. 0.70
SCAN_INTERVAL_SECONDS  -> e.g. 60

# Jupiter public aggregator (no key required)
# you can override these if Jupiter ever changes paths:
JUPITER_QUOTE_URL      -> https://quote-api.jup.ag/v6/quote
JUPITER_SWAP_URL       -> https://quote-api.jup.ag/v6/swap

# safety / misc
PIN                    -> any 4–6 digit number you set (future controls)
NEGATIVE_KEYWORDS      -> comma list, e.g. "honeypot,scam,rugpull"
MEME_TOKENS            -> optional comma list of "NAME:mint" pairs,
                          e.g. "BONK:mint1,WIF:mint2". If empty, defaults below are used.
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

# ---------- optional Solana / solders support ----------
SOLANA_OK = False
try:
    from solana.rpc.api import Client as RpcClient
    from solders.keypair import Keypair

    SOLANA_OK = True
except Exception as exc:
    print(f"[solana] Import error (live trading disabled): {exc!r}")
    SOLANA_OK = False

# ---------- load env ----------
load_dotenv()

# ---------- helpers for env ----------
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

STARTING_SOL = env_float("STARTING_SOL", 0.4)
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 180.0)
DAILY_LOSS_LIMIT_USD = env_float("DAILY_LOSS_LIMIT_USD", 25.0)
MAX_TRADE_RISK_SOL = env_float("MAX_TRADE_RISK_SOL", 0.40)
TAKE_PROFIT_MULT = env_float("TAKE_PROFIT_MULT", 2.0)
STOP_LOSS_MULT = env_float("STOP_LOSS_MULT", 0.5)

MIN_PROBABILITY = env_float("MIN_PROBABILITY", 0.70)
SCAN_INTERVAL_SECONDS = env_int("SCAN_INTERVAL_SECONDS", 60)

# Public Jupiter v6 endpoints (no API key)
JUPITER_QUOTE_URL = os.getenv(
    "JUPITER_QUOTE_URL", "https://quote-api.jup.ag/v6/quote"
).rstrip("/")
JUPITER_SWAP_URL = os.getenv(
    "JUPITER_SWAP_URL", "https://quote-api.jup.ag/v6/swap"
).rstrip("/")

PIN_CODE = os.getenv("PIN", "0000").strip()
NEGATIVE_KEYWORDS: List[str] = [
    k.strip().lower()
    for k in os.getenv("NEGATIVE_KEYWORDS", "").split(",")
    if k.strip()
]

# ---------- aggressive default meme list (EXAMPLES) ----------
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

# ---------- state ----------
balance_sol: float = STARTING_SOL
start_day = datetime.utcnow().date()
realized_pnl_usd: float = 0.0

# SOL mint (wrapped SOL)
SOL_MINT = "So11111111111111111111111111111111111111112"

# ---------- Solana / Jupiter helpers ----------
def init_solana_wallet():
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
    except Exception as exc:
        print(f"[wallet] Error loading wallet: {exc!r}")
        return None, None


def jupiter_quote(
    input_mint: str,
    output_mint: str,
    amount_lamports: int,
    slippage_bps: int = 500,
) -> Optional[Dict[str, Any]]:
    """
    Call Jupiter public quote endpoint.
    Returns a single quote route dict, or None if no route / error.
    """
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount_lamports),
        "slippageBps": str(slippage_bps),
        "onlyDirectRoutes": "false",
    }

    try:
        resp = requests.get(JUPITER_QUOTE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # v6 returns {"data": [route, ...], ...}
        routes = data.get("data") or []
        if not routes:
            print("[swap] quote: no routes in response.")
            return None

        route = routes[0]
        return route

    except requests.exceptions.HTTPError as exc:
        try:
            status = exc.response.status_code
        except Exception:
            status = "?"
        print(f"[swap] quote HTTP {status}; treating as no route.")
        return None
    except Exception as exc:
        print(f"[swap] quote network/error: {exc!r}")
        return None


def jupiter_swap(wallet: Keypair, client: RpcClient, route: Dict[str, Any]) -> Optional[str]:
    """
    Submit a Jupiter swap using /v6/swap.
    On success returns transaction signature; otherwise None.
    """
    try:
        user_pubkey = str(wallet.pubkey())
        payload = {
            "quoteResponse": route,
            "userPublicKey": user_pubkey,
            "wrapAndUnwrapSol": True,
        }

        resp = requests.post(JUPITER_SWAP_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "swapTransaction" not in data:
            print("[swap] swapTransaction missing in response.")
            return None

        swap_tx_b64 = data["swapTransaction"]
        raw_tx = base64.b64decode(swap_tx_b64)

        send_resp = client.send_raw_transaction(raw_tx)
        sig = send_resp.get("result") or send_resp
        print(f"[swap] submitted tx: {sig}")
        return str(sig)
    except requests.exceptions.HTTPError as exc:
        try:
            status = exc.response.status_code
        except Exception:
            status = "?"
        print(f"[swap] swap HTTP {status}; trade aborted.")
        return None
    except Exception as exc:
        print(f"[swap] swap error: {exc!r}")
        return None

# ---------- signal + trading logic ----------
def sample_signal() -> Optional[Dict[str, Any]]:
    """
    Aggressive toy signal:
    - picks random meme token
    - draws probability ~Beta(8,2) (skew high)
    - if prob >= MIN_PROBABILITY, returns signal
    """
    if not MEME_TOKENS:
        return None

    names = list(MEME_TOKENS.keys())
    idx = np.random.randint(0, len(names))
    name = names[idx]
    mint = MEME_TOKENS[name]

    prob = float(np.random.beta(8, 2))
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
        print(f"[risk] daily loss cap hit {realized_pnl_usd:.2f} USD; no more trades today.")
        return True

    return False


def trade_once():
    """One trading cycle: get signal → risk checks → (sim or live) trade."""
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

    trade_size_sol = min(balance_sol * risk_ratio, MAX_TRADE_RISK_SOL)
    if trade_size_sol <= 0:
        print("[trade] No SOL available to risk.")
        return

    trade_size_lamports = int(trade_size_sol * 1_000_000_000)

    # ---------- SIMULATION PATH ----------
    if SIMULATION_MODE or not SOLANA_OK:
        pnl_mult = float(np.random.normal(loc=1.15, scale=0.5))
        new_balance = balance_sol - trade_size_sol + trade_size_sol * pnl_mult
        pnl_sol = new_balance - balance_sol
        balance_sol = new_balance

        pnl_usd = pnl_sol * SOL_PRICE_USD
        realized_pnl_usd += pnl_usd

        print(
            f"[SIM] traded {trade_size_sol:.4f} SOL → pnl={pnl_sol:.4f} SOL, "
            f"new_balance={balance_sol:.4f} SOL, pnl_usd={pnl_usd:.2f}"
        )
        return

    # ---------- LIVE TRADING PATH ----------
    wallet, client = init_solana_wallet()
    if not wallet or not client:
        print("[live] Wallet/client not available; skipping live trade.")
        return

    route = jupiter_quote(SOL_MINT, mint, trade_size_lamports)
    if not route:
        print("[live] No viable route from Jupiter (quote failed).")
        return

    sig = jupiter_swap(wallet, client, route)
    if not sig:
        print("[live] Swap failed; as far as we know, no funds moved.")
        return

    # For now we don't compute true on-chain PnL; just log:
    print(
        f"[LIVE] swap submitted for {trade_size_sol:.4f} SOL into {name} "
        f"(mint={mint}), tx={sig}"
    )

# ---------- main loop ----------
def main():
    print(">>> memebot.py starting")
    print(
        f"[start] SIMULATION_MODE={SIMULATION_MODE} "
        f"STARTING_SOL={STARTING_SOL:.4f} "
        f"SCAN_INTERVAL_SECONDS={SCAN_INTERVAL_SECONDS}"
    )

    if not SOLANA_OK and not SIMULATION_MODE:
        print(
            "[fatal] SIMULATION_MODE=False but SOLANA_OK=False; "
            "install solana + solders or switch SIMULATION_MODE=True."
        )
        return

    if START_MODE == "withdraw":
        print("[withdraw] START_MODE=withdraw not implemented yet; exiting.")
        return

    # schedule periodic trading job
    schedule.every(SCAN_INTERVAL_SECONDS).seconds.do(trade_once)

    # run immediately once on start
    trade_once()

    # main loop
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
