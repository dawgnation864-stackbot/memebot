#!/usr/bin/env python3
"""Memebot - Solana memecoin bot (cloud ready).

Educational example ONLY.
Not financial advice. Extremely high risk. Never trade money you cannot lose.

Default is SIMULATION_MODE=True.
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
from duckduckgo_search import DDGS  # currently unused but kept for future use

# ---------- optional Solana / Jupiter stack ----------
try:
    from solana.keypair import Keypair
    from solana.rpc.api import Client
    from solana.publickey import PublicKey
    from solana.system_program import TransferParams, transfer
    from solders.transaction import VersionedTransaction
    from solders.message import MessageV0
    from solders.instruction import Instruction, AccountMeta
    from solders.pubkey import Pubkey

    SOLANA_OK = True
except Exception:
    SOLANA_OK = False

# ---------- load .env ----------
load_dotenv()

# ---------- configuration from env ----------
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "True").lower() == "true"
START_MODE = os.getenv("START_MODE", "start").lower()  # 'start' or 'withdraw'

STARTING_USD = float(os.getenv("STARTING_USD", "100"))          # challenge start
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "250"))        # rough price
DAILY_LOSS_LIMIT_USD = float(os.getenv("DAILY_LOSS_LIMIT_USD", "25"))

TRADE_THRESHOLD = float(os.getenv("TRADE_THRESHOLD", "0.72"))   # min prob to trade

DB_FILE = os.getenv("DB_FILE", "memebot.db")

JUPITER_ENDPOINT = os.getenv("JUPITER_ENDPOINT", "https://quote-api.jup.ag/v6")
SOLANA_RPC = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com")
WAL
