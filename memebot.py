#!/usr/bin/env python3
"""
Memebot - Solana memecoin bot (cloud ready)
Features:
- GMGN / PumpFun / KOL scanners (best effort)
- Dexscreener lookups
- Daily learning + daily summary
- Tuned trade threshold
- Daily loss limit ($25 default)
- Sell-then-withdraw
"""
IMPORTANT

Educational example. Not financial advice. Use a fresh wallet and small funds.

Live trading needs valid WALLET_PRIVATE_KEY (base58) and WITHDRAWAL_ADDRESS. """


===== Standard library =====

import asyncio import base64 import base58 import os import re import sqlite3 import time from datetime import datetime

===== Third‑party =====

import numpy as np import pandas as pd import requests import schedule from dotenv import load_dotenv

ML (tiny NN). Torch is optional on cloud; we fall back to sigmoid if missing.

try: import torch import torch.nn as nn import torch.optim as optim TORCH_OK = True except Exception: TORCH_OK = False

Optional libs (only used if SIMULATION_MODE=False for live swaps)

try: from solana.keypair import Keypair from solana.publickey import PublicKey from solana.rpc.api import Client from solana.system_program import TransferParams, transfer from solders.transaction import VersionedTransaction from solders.message import MessageV0 from solders.instruction import Instruction, AccountMeta from solders.pubkey import Pubkey SOLANA_OK = True except Exception: SOLANA_OK = False Keypair = None  # type: ignore

===== Config =====

load_dotenv()

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN") GMGN_API_KEY = os.getenv("GMGN_API_KEY")

WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY")  # base58 WITHDRAWAL_ADDRESS = os.getenv("WITHDRAWAL_ADDRESS")  # string address PIN = os.getenv("PIN", "1234")

SIMULATION_MODE = os.getenv("SIMULATION_MODE", "True").lower() == "true" JUPITER_ENDPOINT = "https://quote-api.jup.ag/v6" SOLANA_RPC = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com")

Challenge settings (USD based)

STARTING_USD = float(os.getenv("STARTING_USD", "100")) SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "250")) DAILY_LOSS_LIMIT_USD = float(os.getenv("DAILY_LOSS_LIMIT_USD", "25")) STARTING_CAPITAL_SOL = STARTING_USD / max(SOL_PRICE_USD, 1e-6) DAILY_LOSS_LIMIT_SOL = DAILY_LOSS_LIMIT_USD / max(SOL_PRICE_USD, 1e-6)

Trading threshold (be picky)

TRADE_THRESHOLD = float(os.getenv("TRADE_THRESHOLD", "0.72"))

TRADE_AMOUNT_LAMPORTS = int(os.getenv("TRADE_AMOUNT_LAMPORTS", "100000000"))  # 0.1 SOL MODEL_PATH = "memecoin_model.pth" DB_FILE = os.getenv("DB_FILE", "memecoins.db")

NEGATIVE_KEYWORDS = ( "scam OR rug OR honeypot OR dump OR fake OR phishing OR rugpull" ) KOL_HANDLES = [ "A1lon9", "aeyakovenko", "Blknoiz06", "DegenerateNews", "Theunipcs", "Kmoney_69", "CryptoWendyO", "lmrankhan", "TheCryptoLark", "MattWallace888", ]

===== Database =====

def init_db(): conn = sqlite3.connect(DB_FILE) c = conn.cursor() c.execute( """ CREATE TABLE IF NOT EXISTS memecoins ( symbol TEXT, name TEXT, contract_address TEXT PRIMARY KEY, market_cap REAL DEFAULT 0, volume_1h REAL DEFAULT 0, holders INTEGER DEFAULT 0, liquidity REAL DEFAULT 0, mentions_x INTEGER DEFAULT 0, negative_mentions INTEGER DEFAULT 0, risk_ratio REAL DEFAULT 0.0, source TEXT DEFAULT 'unknown', last_updated TIMESTAMP ) """ ) c.execute( """ CREATE TABLE IF NOT EXISTS history ( id INTEGER PRIMARY KEY AUTOINCREMENT, contract_address TEXT, timestamp TIMESTAMP, market_cap REAL, volume_1h REAL, holders INTEGER, liquidity REAL, mentions_x INTEGER, negative_mentions INTEGER, risk_ratio REAL, holder_growth REAL DEFAULT 0, volume_growth REAL DEFAULT 0, mention_velocity REAL DEFAULT 0, liq_ratio REAL DEFAULT 0, success_label INTEGER DEFAULT 0, FOREIGN KEY (contract_address) REFERENCES memecoins (contract_address) ) """ ) c.execute( """ CREATE TABLE IF NOT EXISTS trades ( id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, contract_address TEXT, trade_time TIMESTAMP, action TEXT, amount_lamports REAL, token_amount REAL, pnl_sol REAL DEFAULT 0, outcome TEXT ) """ ) c.execute( """ CREATE TABLE IF NOT EXISTS portfolio ( id INTEGER PRIMARY KEY AUTOINCREMENT, date TIMESTAMP, capital_sol REAL, daily_pnl_sol REAL, trades_today INTEGER ) """ ) c.execute( """ CREATE TABLE IF NOT EXISTS learn_events ( id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TIMESTAMP, symbol TEXT, contract_address TEXT, prob REAL, risk_ratio REAL, trade_size_sol REAL, pnl_sol REAL, label INTEGER ) """ ) conn.commit() conn.close()

===== ML =====

class MemecoinPredictor: """Tiny wrapper: uses torch if available, else a simple logistic model. Input features: [prob, risk, hour_sin, hour_cos, size_norm, pnl_roll] """ def init(self, input_size=6): self.use_torch = TORCH_OK if TORCH_OK: self.model = nn.Sequential( nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid(), ) if os.path.exists(MODEL_PATH): try: self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")) self.model.eval() print("Loaded existing model.") except Exception: print("Model load failed; starting fresh.") else: # simple weights for logistic fallback self.w = np.zeros((input_size,)) self.b = 0.0

def predict(self, x: np.ndarray) -> float:
    x = x.astype(np.float32)
    if self.use_torch:
        with torch.no_grad():
            t = torch.tensor(x).unsqueeze(0)
            p = self.model(t).item()
        return float(p)
    # logistic fallback
    z = float(np.dot(self.w, x) + self.b)
    return 1.0 / (1.0 + np.exp(-z))

def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 1e-3):
    if X.size == 0:
        return
    if self.use_torch:
        m = self.model
        opt = optim.Adam(m.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        m.train()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad()
            out = m(Xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()
        torch.save(m.state_dict(), MODEL_PATH)
        print(f"[learn] Trained on {len(X)} events.")
    else:
        # simple gradient step for fallback (not great, but keeps example working)
        for _ in range(epochs):
            z = X @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = ((p - y)[:, None] * X).mean(axis=0)
            grad_b = (p - y).mean()
            self.w -= lr * grad_w
            self.b -= lr * grad_b

def load_or_create_model() -> MemecoinPredictor: return MemecoinPredictor()

===== Wallet / Jupiter (live mode) =====

def init_solana_wallet(): if SIMULATION_MODE: print("SIMULATION_MODE=True — no wallet init.") return None if not SOLANA_OK: print("Solana libs not available. Cannot trade live.") return None if not WALLET_PRIVATE_KEY: print("WALLET_PRIVATE_KEY not set.") return None try: secret_key_bytes = base58.b58decode(WALLET_PRIVATE_KEY) wallet = Keypair.from_secret_key(secret_key_bytes) connection = Client(SOLANA_RPC) print(f"Live wallet: {wallet.public_key}") return wallet, connection except Exception as e: print(f"Wallet init error: {e}") return None

def get_quote(input_mint: str, output_mint: str, amount: int) -> dict: url = f"{JUPITER_ENDPOINT}/quote" resp = requests.get(url, params={"inputMint": input_mint, "outputMint": output_mint, "amount": amount}, timeout=20) if resp.status_code == 200: return resp.json() raise RuntimeError(f"Quote error: {resp.text}")

def instruction_to_solana_ix(jup_ix: dict): program_id = Pubkey.from_string(jup_ix["programId"])  # type: ignore keys = [ AccountMeta(pubkey=Pubkey.from_string(acc["pubkey"]), is_signer=acc["isSigner"], is_writable=acc["isWritable"])  # type: ignore for acc in jup_ix["accounts"] ] data = base64.b64decode(jup_ix["data"]) return Instruction(program_id, keys, data)

async def execute_swap(quote: dict, wallet, connection, token_mint: str, amount_lamports: int, is_buy: bool = True): swap_url = f"{JUPITER_ENDPOINT}/swap" req = {"quoteResponse": quote, "userPublicKey": str(wallet.public_key), "wrapAndUnwrapSol": True, "prioritizationFeeLamports": "auto"} r = requests.post(swap_url, json=req, timeout=30) if r.status_code != 200: raise RuntimeError(f"Swap error: {r.text}") swap_tx_buf = base64.b64decode(r.json()["swapTransaction"])  # type: ignore swap_tx = VersionedTransaction.from_bytes(swap_tx_buf) swap_tx.sign([wallet]) txid = connection.send_raw_transaction(bytes(swap_tx), opts={"skip_preflight": True, "max_retries": 2}) print(f"Swap TX: {txid.value}") conf = connection.confirm_transaction(txid.value, commitment="confirmed") if getattr(conf.value, "err", None): raise RuntimeError(f"Swap failed: {conf.value.err}") print(f"Swap confirmed! {'Bought' if is_buy else 'Sold'} {amount_lamports/1e9:.4f} SOL of {token_mint}") return txid.value

def place_live_swap(wallet, connection, token_mint: str, amount_lamports: int, symbol: str, ca: str, is_buy: bool = True): sol_mint = "So11111111111111111111111111111111111111112" inp = sol_mint if is_buy else token_mint outp = token_mint if is_buy else sol_mint try: quote = get_quote(inp, outp, amount_lamports) txid = asyncio.run(execute_swap(quote, wallet, connection, token_mint, amount_lamports, is_buy)) log_trade(symbol, ca, "BUY" if is_buy else "SELL", amount_lamports, 0, 0) return txid except Exception as e: print(f"Live swap error: {e}") return None

===== Trade log, portfolio, learn rows =====

def log_trade(symbol, ca, action, amount_lamports, token_amount=0, pnl=0, outcome="OPEN"): conn = sqlite3.connect(DB_FILE) c = conn.cursor() c.execute( """ INSERT INTO trades (symbol, contract_address, trade_time, action, amount_lamports, token_amount, pnl_sol, outcome) VALUES (?, ?, ?, ?, ?, ?, ?, ?) """, (symbol, ca, datetime.now(), action, amount_lamports, token_amount, pnl, outcome), ) conn.commit() conn.close()

def _close_open_trade(contract_address: str): conn = sqlite3.connect(DB_FILE) c = conn.cursor() c.execute("UPDATE trades SET outcome='CLOSED' WHERE contract_address=? AND outcome='OPEN'", (contract_address,)) conn.commit() conn.close()

def log_portfolio(capital_sol, daily_pnl_sol, trades_today): conn = sqlite3.connect(DB_FILE) c = conn.cursor() conn.commit() conn.close()c.execute("""INSERT INTO learn_events (timestamp, symbol, contract_address, prob, risk_ratio, trade_size_sol, pnl_sol, label)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
          (ts, symbol, ca, float(prob), float(risk_ratio), float(trade_size_sol), float(pnl_sol), int(label)))

def _insert_learn_event(ts, symbol, ca, prob, risk_ratio, trade_size_sol, pnl_sol): conn = sqlite3.connect(DB_FILE) c = conn.cursor() label = 1 if pnl_sol > 0 else 0 c.execute( """ INSERT INTO learn_events (timestamp, symbol, contract_address, prob, risk_ratio, trade_size_sol, pnl_sol, label) VALUES (?, ?, ?, ?, ?, ?, ?, ?) """, (ts, symbol, ca, float(prob), float(risk_ratio), float(trade_size_sol), float(pnl_sol), int(label)), ) conn.commit() conn.close()

def get_today_pnl_sol() -> float: conn = sqlite3.connect(DB_FILE) df = pd.read_sql_query( "SELECT pnl_sol, timestamp FROM learn_events WHERE DATE(timestamp)=DATE('now', 'localtime')", conn, parse_dates=["timestamp"], ) conn.close() return float(df["pnl_sol"].sum()) if not df.empty else 0.0

===== Data sources =====

_SOLANA_MINT_RE = re.compile(r"[1-9A-HJ-NP-Za-km-z]{32,44}")

def _get_json(url, params=None, headers=None, timeout=20): try: r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout) if r.status_code == 200: return r.json() print(f"[data] {url} -> {r.status_code}") except Exception as e: print(f"[data] GET error {url}: {e}") return None

def fetch_top_memecoins_from_gmgn(limit: int = 10): url = "https://gmgn.ai/api/v1/token/sol/trending" headers = {"Authorization": f"Bearer {GMGN_API_KEY}"} if GMGN_API_KEY else {} data = _get_json(url, headers=headers) out = [] if isinstance(data, dict) and "data" in data: for row in data.get("data", [])[:limit]: symbol = row.get("symbol") or row.get("name") or "GMGN" mint = row.get("address") or row.get("mint") if mint: out.append((symbol, mint)) return out

def fetch_trending_from_pumpfun(limit: int = 10): url = "https://pump.fun/api/trending" data = _get_json(url) out = [] if isinstance(data, list): for row in data[:limit]: symbol = row.get("symbol") or row.get("name") or "PUMP" mint = row.get("mint") or row.get("address") if mint: out.append((symbol, mint)) return out

def scan_kol_tweets(max_per_user: int = 20): if not TWITTER_BEARER_TOKEN: return [] out = [] try: import tweepy client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True) for handle in KOL_HANDLES: try: user = client.get_user(username=handle).data if not user: continue tweets = client.get_users_tweets(user.id, max_results=min(20, max_per_user), tweet_fields=["text"]).data or [] for t in tweets: for m in _SOLANA_MINT_RE.findall(t.text or ""): out.append(("KOL", m)) except Exception as e: print(f"[kol] {handle}: {e}") except Exception as e: print(f"[kol] init: {e}") return out

def get_token_info_from_dexscreener(mint: str): url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}" data = _get_json(url) if not data or not isinstance(data, dict): return {} pairs = data.get("pairs") or [] if not pairs: return {} p = pairs[0] return { "symbol": p.get("baseToken", {}).get("symbol"), "fdv": p.get("fdv"), "liquidity": (p.get("liquidity", {}) or {}).get("usd"), "volume24h": p.get("volume", {}).get("h24"), "priceUsd": p.get("priceUsd"), }

===== Strategy pieces =====

def simulate_trade(capital_sol, signal_strength, risk_ratio): # Simple asymmetric returns, capped loss if risk_ratio > 0.3: loss = 0.1 return capital_sol - loss, -loss ret = 0.862 if signal_strength > 0.5 else -0.2 gain = capital_sol * ret if gain < -0.1: gain = -0.1 return capital_sol + gain, gain

def _predict_success(model: MemecoinPredictor, features: np.ndarray) -> float: return model.predict(features)

def analyze_token_success(wallet_info, model, current_capital_sol: float = STARTING_CAPITAL_SOL): print("Analyzing token success… (learn-enabled, tuned threshold)")

# 1) candidates
cands = []
try:
    cands += [(s or "GMGN", m, "gmgn") for s, m in fetch_top_memecoins_from_gmgn(limit=8)]
except Exception as e:
    print(f"[gmgn] {e}")
try:
    cands += [(s or "PUMP", m, "pump") for s, m in fetch_trending_from_pumpfun(limit=8)]
except Exception as e:
    print(f"[pumpfun] {e}")
try:
    cands += [(s or "KOL", m, "kol") for s, m in scan_kol_tweets(max_per_user=10)]
except Exception as e:
    print(f"[kol] {e}")

# dedupe by mint
seen, uniq = set(), []
for sym, m, src in cands:
    if not m or m in seen:
        continue
    seen.add(m)
    uniq.append((sym, m, src))
if not uniq:
    uniq = [("DUMMY", "So11111111111111111111111111111111111111112", "fallback")]

trades_today = 0
daily_pnl_sol = 0.0

now = datetime.now()
hour_sin = np.sin(2 * np.pi * now.hour / 24.0)
hour_cos = np.cos(2 * np.pi * now.hour / 24.0)

for symbol, ca, src in uniq[:3]:
    info = get_token_info_from_dexscreener(ca)
    liq = (info.get("liquidity") or 0) or 0
    vol24 = (info.get("volume24h") or 0) or 0

    # Heuristic base prob
    base_prob = 0.55
    if src == "kol":
        base_prob += 0.1
    if liq and liq > 10000:
        base_prob += 0.05
    if vol24 and vol24 > 50000:
        base_prob += 0.05
    base_prob = float(max(0.0, min(0.95, base_prob)))

    risk_ratio = 0.25
    if liq and liq < 3000:
        risk_ratio += 0.1

    # Daily loss stop
    today_pnl = get_today_pnl_sol()
    if today_pnl <= -DAILY_LOSS_LIMIT_SOL:
        print(f"[risk] Daily loss limit hit ({today_pnl:.4f} SOL). Skipping new trades today.")
        break

    trade_amount_lamports = int(current_capital_sol * 1e9)
    trade_size_sol = trade_amount_lamports / 1e9

    features = np.array([base_prob, risk_ratio, hour_sin, hour_cos, trade_size_sol, 0.0], dtype=float)
    pred_p = _predict_success(model, features)

    blended = 0.5 * base_prob + 0.5 * pred_p
    do_trade = blended >= TRADE_THRESHOLD

    if SIMULATION_MODE and do_trade:
        new_cap, _gain = simulate_trade(current_capital_sol, base_prob, risk_ratio)
        pnl = new_cap - current_capital_sol
        daily_pnl_sol += pnl
        current_capital_sol = new_cap
        trades_today += 1
        _insert_learn_event(datetime.now(), symbol, ca, base_prob, risk_ratio, trade_size_sol, pnl)
    elif not SIMULATION_MODE and do_trade:
        if wallet_info:
            _ = place_live_swap(wallet_info[0], wallet_info[1], ca, trade_amount_lamports, symbol, ca, is_buy=True)
        trades_today += 1
        _insert_learn_event(datetime.now(), symbol, ca, base_prob, risk_ratio, trade_size_sol, 0.0)

log_portfolio(current_capital_sol, daily_pnl_sol, trades_today)
print(f"End of cycle — capital: {current_capital_sol:.4f} SOL | daily PnL: {daily_pnl_sol:.4f} SOL | trades: {trades_today}")

===== Reporting & learning orchestration =====

def update_labels(): pass  # labels created at insert time

def _build_training_dataframe(): conn = sqlite3.connect(DB_FILE) df = pd.read_sql_query("SELECT * FROM learn_events ORDER BY timestamp ASC", conn, parse_dates=["timestamp"])  # type: ignore conn.close() if df.empty: return df df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24.0) df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24.0) df["size_norm"] = df["trade_size_sol"].clip(1e-6, None) df["pnl_roll"] = ( df.sort_values("timestamp").groupby("symbol")["pnl_sol"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True) ) feat = ["prob", "risk_ratio", "hour_sin", "hour_cos", "size_norm", "pnl_roll"] df[feat] = df[feat].astype(float).fillna(0.0) return df

def train_model(): df = _build_training_dataframe() if df.empty or df["label"].sum() == 0: print("[learn] Not enough data to train yet.") if TORCH_OK and not os.path.exists(MODEL_PATH): torch.save(MemecoinPredictor().model.state_dict(), MODEL_PATH)  # type: ignore return X = df[["prob", "risk_ratio", "hour_sin", "hour_cos", "size_norm", "pnl_roll"]].values y = df["label"].values.astype(float) model = MemecoinPredictor(input_size=X.shape[1]) model.train(X, y, epochs=200, lr=1e-3)

def daily_summary(): conn = sqlite3.connect(DB_FILE) df = pd.read_sql_query("SELECT * FROM learn_events WHERE DATE(timestamp)=DATE('now', 'localtime')", conn, parse_dates=["timestamp"])  # type: ignore conn.close() if df.empty: print("[summary] No learn events today yet.") return total = len(df) wins = int((df["pnl_sol"] > 0).sum()) pnl = float(df["pnl_sol"].sum()) wr = (wins / total) * 100.0 print(f"[summary] Today: trades={total}, wins={wins} ({wr:.1f}%), pnl={pnl:.4f} SOL")

===== Emergency: sell then withdraw =====

def emergency_withdrawal(wallet, connection): pin_input = input("Enter PIN to stop bot and withdraw funds: ") if pin_input != PIN: print("Invalid PIN. Withdrawal denied.") return print("PIN ok. Try sell‑then‑withdraw…")

try:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT contract_address, symbol FROM trades WHERE outcome='OPEN' ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        ca, sym = row[0], row[1]
        try:
            place_live_swap(wallet, connection, ca, int(0.01 * 1e9), sym, ca, is_buy=False)
            _close_open_trade(ca)
        except Exception as e:
            print(f"Sell attempt failed: {e}")
    else:
        print("No OPEN trades recorded.")
except Exception as e:
    print(f"Open trade check error: {e}")

if not WITHDRAWAL_ADDRESS:
    print("WITHDRAWAL_ADDRESS not set.")
    return
try:
    bal = connection.get_balance(wallet.public_key).value / 1e9
    if bal > 0.01:
        ix = transfer(TransferParams(from_pubkey=wallet.public_key, to_pubkey=PublicKey(WITHDRAWAL_ADDRESS), lamports=int((bal - 0.005) * 1e9)))
        recent = connection.get_latest_blockhash().value.blockhash
        msg = MessageV0.try_compile(payer=wallet.public_key, instructions=[ix], address_lookup_table_accounts=[], recent_blockhash=recent)
        tx = VersionedTransaction(msg, [wallet])
        txid = connection.send_raw_transaction(bytes(tx))
        conf = connection.confirm_transaction(txid.value)
        if not getattr(conf.value, "err", None):
            print(f"Withdrew {bal:.4f} SOL to {WITHDRAWAL_ADDRESS}. TX: {txid.value}")
        else:
            print("Withdrawal failed.")
except Exception as e:
    print(f"Withdrawal error: {e}")
print("Bot stopped.")
raise SystemExit(0)

===== Scheduler =====

def job(): print(f"Job start @ {datetime.now().isoformat(timespec='seconds')}") model = load_or_create_model() wallet_info = init_solana_wallet() if not SIMULATION_MODE else None analyze_token_success(wallet_info, model)

def main(): init_db()

# Optional one‑shot withdrawal on boot
if os.getenv("START_MODE", "start").lower() == "withdraw" and not SIMULATION_MODE and SOLANA_OK:
    wi = init_solana_wallet()
    if wi:
        emergency_withdrawal(wi[0], wi[1])
    return

schedule.every(1).hours.do(job)
schedule.every().day.at("02:00").do(lambda: (update_labels(), train_model()))
schedule.every().day.at("23:59").do(daily_summary)

job()  # run once now

try:
    while True:
        schedule.run_pending()
        time.sleep(60)
except KeyboardInterrupt:
    print("Exiting…")

if name == "main": main()
