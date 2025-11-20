"""
Memebot – Solana memecoin bot using Jupiter v6

Key points:
- Uses Jupiter v6 HTTP API (quote + swap) for swaps.
- Uses solders only (no solana-py).
- Can run in simulation or live mode, controlled by SIMULATION_MODE env var.
- Reads all configuration from environment variables (Railway service variables).

Environment variables (all strings):

  SIMULATION_MODE        -> "True" (default) or "False"
  START_MODE             -> "start" (default) or "withdraw"

  STARTING_SOL           -> e.g. "0.4000"
  STARTING_USD           -> e.g. "100"
  DAILY_LOSS_LIMIT_USD   -> e.g. "25"
  MAX_TRADE_RISK_SOL     -> e.g. "0.4"
  MIN_PROBABILITY        -> e.g. "0.72"
  STOP_LOSS_MULT         -> e.g. "0.97"
  TAKE_PROFIT_MULT       -> e.g. "1.05"
  SCAN_INTERVAL_SECONDS  -> e.g. "60"

  SOLANA_RPC             -> e.g. "https://api.mainnet-beta.solana.com"
  JUPITER_ENDPOINT       -> e.g. "https://quote-api.jup.ag"

  MEME_TOKENS            -> comma-separated list of mint addresses to trade
                            e.g. "So11111111111111111111111111111111111111112"

  WALLET_PRIVATE_KEY     -> base58 encoded private key (Phantom export format)
  WITHDRAWAL_ADDRESS     -> base58 Solana address (for future withdraw logic)
  PIN                    -> arbitrary string, not used in core logic yet
"""

from __future__ import annotations

# ---------- standard library ----------
import os
import asyncio
import base64
import time
import random
from datetime import datetime, date
from typing import Optional, Tuple, List

# ---------- third-party ----------
import requests
import numpy as np  # used for simple risk / probability calcs
from dotenv import load_dotenv

# ---------- optional Solana / solders stack ----------
SOLANA_OK = False
try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.rpc.api import Client as RpcClient
    from solders.transaction import VersionedTransaction
    from solders import message

    SOLANA_OK = True
except Exception as exc:  # noqa: BLE001
    print(f"[solana] Import error: {exc!r}")
    SOLANA_OK = False
    Keypair = Pubkey = RpcClient = VersionedTransaction = message = None  # type: ignore

# ---------- load .env (local dev) ----------
load_dotenv()

# ---------- configuration helpers ----------


def env_bool(name: str, default: str = "True") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y")


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:  # noqa: BLE001
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:  # noqa: BLE001
        return default


SIMULATION_MODE: bool = env_bool("SIMULATION_MODE", "True")
START_MODE: str = os.getenv("START_MODE", "start").strip().lower()

STARTING_SOL: float = env_float("STARTING_SOL", 0.4)
STARTING_USD: float = env_float("STARTING_USD", 100.0)
DAILY_LOSS_LIMIT_USD: float = env_float("DAILY_LOSS_LIMIT_USD", 25.0)

MAX_TRADE_RISK_SOL: float = env_float("MAX_TRADE_RISK_SOL", STARTING_SOL)
MIN_PROBABILITY: float = env_float("MIN_PROBABILITY", 0.72)
STOP_LOSS_MULT: float = env_float("STOP_LOSS_MULT", 0.97)
TAKE_PROFIT_MULT: float = env_float("TAKE_PROFIT_MULT", 1.05)

SCAN_INTERVAL_SECONDS: int = env_int("SCAN_INTERVAL_SECONDS", 60)

SOLANA_RPC: str = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com")
JUPITER_ENDPOINT: str = os.getenv("JUPITER_ENDPOINT", "https://quote-api.jup.ag")

WALLET_PRIVATE_KEY: str = os.getenv("WALLET_PRIVATE_KEY", "").strip()
WITHDRAWAL_ADDRESS: str = os.getenv("WITHDRAWAL_ADDRESS", "").strip()
PIN: str = os.getenv("PIN", "").strip()

MEME_TOKENS_ENV: str = os.getenv("MEME_TOKENS", "").strip()
MEME_TOKENS: List[str] = [
    mint.strip()
    for mint in MEME_TOKENS_ENV.split(",")
    if mint.strip()
]

# Jupiter uses wrapped SOL mint:
WSOL_MINT = "So11111111111111111111111111111111111111112"

# ---------- runtime state ----------

portfolio_sol: float = STARTING_SOL
starting_day: date = date.today()
daily_realized_pnl_usd: float = 0.0


# ---------- wallet helpers ----------


def init_solana_wallet() -> Tuple[Optional["Keypair"], Optional["RpcClient"]]:
    """Load wallet from WALLET_PRIVATE_KEY and set up RPC client."""
    if not SOLANA_OK:
        print("[wallet] SOLANA_OK=False (solders imports missing).")
        return None, None

    if not WALLET_PRIVATE_KEY:
        print("[wallet] WALLET_PRIVATE_KEY env not set.")
        return None, None

    try:
        import base58  # local import, installed via requirements

        secret_key_bytes = base58.b58decode(WALLET_PRIVATE_KEY)
        wallet = Keypair.from_secret_key(secret_key_bytes)  # type: ignore[arg-type]
        client = RpcClient(SOLANA_RPC)
        print(f"[wallet] Loaded wallet: {wallet.pubkey()}")
        return wallet, client
    except Exception as exc:  # noqa: BLE001
        print(f"[wallet] Error loading wallet: {exc!r}")
        return None, None


# ---------- Jupiter helpers ----------


def jupiter_quote(
    input_mint: str,
    output_mint: str,
    amount_lamports: int,
    slippage_bps: int = 50,
) -> Optional[dict]:
    """Get a quote from Jupiter v6."""
    try:
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount_lamports),
            "slippageBps": str(slippage_bps),
        }
        url = f"{JUPITER_ENDPOINT}/v6/quote"
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        quote = resp.json()
        return quote
    except Exception as exc:  # noqa: BLE001
        print(f"[jupiter] Quote error: {exc!r}")
        return None


def jupiter_swap(
    wallet: "Keypair",
    client: "RpcClient",
    quote: dict,
) -> Optional[str]:
    """
    Execute a Jupiter swap from a quote.

    Returns signature string if successful, else None.
    """
    try:
        url = f"{JUPITER_ENDPOINT}/v6/swap"
        body = {
            "quoteResponse": quote,
            "userPublicKey": str(wallet.pubkey()),
            "wrapAndUnwrapSol": True,
        }
        resp = requests.post(url, json=body, timeout=20)
        resp.raise_for_status()
        swap_tx_b64 = resp.json()["swapTransaction"]

        raw_tx = VersionedTransaction.from_bytes(base64.b64decode(swap_tx_b64))  # type: ignore[arg-type]
        # Sign with wallet
        tx_bytes = message.to_bytes_versioned(raw_tx.message)  # type: ignore[attr-defined]
        sig = wallet.sign_message(tx_bytes)
        signed_tx = VersionedTransaction.populate(raw_tx.message, [sig])  # type: ignore[arg-type]

        tx_sig = client.send_raw_transaction(bytes(signed_tx))
        print(f"[swap] Sent transaction: {tx_sig}")
        return str(tx_sig)
    except Exception as exc:  # noqa: BLE001
        print(f"[swap] Error sending swap: {exc!r}")
        return None


# ---------- strategy / risk helpers ----------


def reset_daily_pnl_if_needed() -> None:
    global starting_day, daily_realized_pnl_usd
    today = date.today()
    if today != starting_day:
        print("[risk] New day detected, resetting daily PnL.")
        starting_day = today
        daily_realized_pnl_usd = 0.0


def can_trade_today(sol_price_usd: float) -> bool:
    reset_daily_pnl_if_needed()
    if DAILY_LOSS_LIMIT_USD <= 0:
        return True
    if daily_realized_pnl_usd <= -DAILY_LOSS_LIMIT_USD:
        print(
            f"[risk] Daily loss limit hit: {daily_realized_pnl_usd:.2f} USD "
            f"<= -{DAILY_LOSS_LIMIT_USD:.2f} USD",
        )
        return False
    return True


def estimate_sol_price_usd() -> float:
    """Rough SOL price from starting values, fallback to 200."""
    if STARTING_SOL > 0 and STARTING_USD > 0:
        return STARTING_USD / STARTING_SOL
    return 200.0


def choose_trade_size() -> float:
    """How much SOL to risk on this trade."""
    global portfolio_sol
    amount = min(MAX_TRADE_RISK_SOL, portfolio_sol)
    amount = max(0.0, amount)
    return amount


def pick_meme_token() -> Optional[str]:
    """Pick a token mint from MEME_TOKENS env list."""
    if not MEME_TOKENS:
        print("[signal] No MEME_TOKENS configured; skipping live trade.")
        return None
    return random.choice(MEME_TOKENS)


# ---------- simulation logic (no real trades) ----------


def run_simulated_trade() -> None:
    """Simple fake trade to exercise logic."""
    global portfolio_sol, daily_realized_pnl_usd

    sol_price = estimate_sol_price_usd()
    if not can_trade_today(sol_price):
        return

    prob = float(np.clip(np.random.normal(loc=0.8, scale=0.05), 0.0, 1.0))
    print(f"[signal] DEMO prob={prob:.3f} threshold={MIN_PROBABILITY:.3f}")
    if prob < MIN_PROBABILITY:
        print("[SIM] No trade (probability below threshold).")
        return

    size_sol = choose_trade_size()
    if size_sol <= 0:
        print("[SIM] No trade; size <= 0.")
        return

    # Simulate PnL using simple random RR
    rr = np.random.choice([TAKE_PROFIT_MULT, STOP_LOSS_MULT], p=[0.55, 0.45])
    new_value = portfolio_sol * rr
    pnl_sol = new_value - portfolio_sol
    portfolio_sol = new_value

    pnl_usd = pnl_sol * sol_price
    daily_realized_pnl_usd += pnl_usd

    print(
        f"[SIM] traded {size_sol:.4f} SOL  ->  pnl={pnl_sol:.4f} SOL "
        f"({pnl_usd:+.2f} USD), new_portfolio={portfolio_sol:.4f} SOL",
    )


# ---------- live trading logic ----------


def run_live_trade(wallet: "Keypair", client: "RpcClient") -> None:
    """
    Live mode: pick a meme token and try one Jupiter swap SOL -> token.

    NOTE: PnL tracking is approximate; we do not query on-chain balances here.
    """
    global portfolio_sol

    sol_price = estimate_sol_price_usd()
    if not can_trade_today(sol_price):
        return

    token_mint = pick_meme_token()
    if not token_mint:
        return

    size_sol = choose_trade_size()
    if size_sol <= 0:
        print("[LIVE] No trade; size <= 0.")
        return

    lamports = int(size_sol * 1_000_000_000)
    print(
        f"[LIVE] Considering trade: {size_sol:.4f} SOL "
        f"({lamports} lamports) -> token {token_mint[:6]}...",
    )

    quote = jupiter_quote(WSOL_MINT, token_mint, lamports, slippage_bps=50)
    if not quote:
        print("[LIVE] No quote returned; skipping.")
        return

    # Very basic sanity check on quote
    out_amount = float(quote.get("outAmount", 0)) / (10 ** quote.get("outDecimals", 6))
    if out_amount <= 0:
        print("[LIVE] Quote outAmount <= 0; skipping.")
        return

    print(
        f"[LIVE] Quote ok: in={size_sol:.4f} SOL -> "
        f"out≈{out_amount:.6f} ({token_mint[:6]}...)",
    )

    sig = jupiter_swap(wallet, client, quote)
    if sig:
        portfolio_sol -= size_sol
        print(
            f"[LIVE] Swap submitted, approx new_portfolio={portfolio_sol:.4f} SOL "
            f"(tx={sig})",
        )
    else:
        print("[LIVE] Swap failed; portfolio unchanged.")


# ---------- main loop ----------


async def main_loop() -> None:
    print(">>> memebot.py started (top of file reached)")
    print(
        f"[start] Memebot | SIM={SIMULATION_MODE} | START={STARTING_SOL:.4f} SOL "
        f"| MODE={START_MODE}",
    )

    wallet: Optional["Keypair"] = None
    client: Optional["RpcClient"] = None

    if not SIMULATION_MODE:
        if not SOLANA_OK:
            print(
                "[fatal] SIMULATION_MODE=False but SOLANA_OK=False; "
                "cannot go live until solders is installed correctly.",
            )
            return

        wallet, client = init_solana_wallet()
        if not wallet or not client:
            print(
                "[fatal] SIMULATION_MODE=False but wallet/client not available; "
                "cannot go live.",
            )
            return

    # START_MODE: for now we only support normal "start"
    if START_MODE != "start":
        print(
            f"[start] START_MODE={START_MODE!r} – withdraw logic "
            "not implemented yet; exiting safely.",
        )
        return

    # main loop
    while True:
        try:
            print("[job] Fetching signals…")
            if SIMULATION_MODE:
                run_simulated_trade()
            else:
                assert wallet is not None and client is not None
                run_live_trade(wallet, client)

        except Exception as exc:  # noqa: BLE001
            print(f"[error] Unhandled exception in main loop: {exc!r}")

        # Wait until next scan
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt – shutting down memebot.")
