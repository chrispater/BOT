import os
import asyncio
import threading
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import json
from datetime import datetime
from pathlib import Path

from .database import (
    init_db, get_db, save_trade, save_signal, save_performance,
    get_user_trades, get_user_signals, get_latest_performance,
    get_user_settings, update_user_settings,
    save_optimization_job, update_optimization_job, get_optimization_job,
    try_start_optimization_job, save_optimization_run, get_optimization_runs,
    get_optimization_run, get_latest_optimization_run,
    get_user_permissions, get_pending_users, get_all_users,
    update_user_status, update_user_permissions, make_user_admin
)
from .auth import (
    get_password_hash, verify_password, create_access_token,
    decode_token, encrypt_credential, decrypt_credential
)
from .trading_service import TradingService, ParameterOptimizer

app = FastAPI(title="Crypto Trading Bot API", version="1.0.0")
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_bots: Dict[int, TradingService] = {}
user_connections: Dict[int, list] = {}
optimization_jobs: Dict[int, dict] = {}

class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class APICredentials(BaseModel):
    api_key: str
    api_secret: str
    api_password: str

class TradingSettings(BaseModel):
    starting_balance: Optional[float] = None
    leverage: Optional[int] = None
    selected_coins: Optional[List[str]] = None
    risk_per_trade: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trade_cooldown: Optional[int] = None
    min_confidence: Optional[float] = None
    timeframe: Optional[str] = None
    # Compounding & risk enhancement params
    trailing_stop_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    retrain_every: Optional[int] = None
    profit_risk_multiplier: Optional[float] = None

VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']

AVAILABLE_COINS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'DOGE/USDT:USDT', 'BNB/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT',
    'LINK/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT', 'UNI/USDT:USDT',
    'SHIB/USDT:USDT', 'LTC/USDT:USDT', 'ATOM/USDT:USDT'
]

# Timeframe → seconds mapping used to align bot loop sleep with candle duration
_TF_SECONDS = {
    '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
    '1h': 3600, '2h': 7200, '4h': 14400, '1d': 86400
}

def _bot_sleep_seconds(timeframe: str) -> float:
    """Sleep 85% of the candle duration so the cycle fires slightly before each close."""
    return max(30, _TF_SECONDS.get(timeframe, 300) * 0.85)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload

@app.on_event("startup")
async def startup():
    init_db()

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/auth/register")
async def register(user: UserRegister):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = %s", (user.username,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Username exists")
        password_hash = get_password_hash(user.password)
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (user.username, password_hash)
        )
        user_id = cur.fetchone()['id']
        conn.commit()
    perms = get_user_permissions(user_id)
    token = create_access_token({"sub": user.username, "user_id": user_id})
    return {
        "access_token": token,
        "token_type": "bearer",
        "permissions": perms
    }

@app.post("/api/auth/login")
async def login(user: UserLogin):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (user.username,))
        row = cur.fetchone()
        if not row or not verify_password(user.password, row['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid credentials")
    perms = get_user_permissions(row['id'])
    token = create_access_token({"sub": user.username, "user_id": row['id']})
    return {
        "access_token": token,
        "token_type": "bearer",
        "permissions": perms
    }

@app.post("/api/credentials")
async def save_credentials(creds: APICredentials, user = Depends(get_current_user)):
    user_id = user['user_id']
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users SET
                encrypted_api_key = %s,
                encrypted_api_secret = %s,
                encrypted_api_password = %s
            WHERE id = %s
        """, (
            encrypt_credential(creds.api_key),
            encrypt_credential(creds.api_secret),
            encrypt_credential(creds.api_password),
            user_id
        ))
        conn.commit()
    return {"status": "credentials saved"}

@app.get("/api/credentials/status")
async def credentials_status(user = Depends(get_current_user)):
    user_id = user['user_id']
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT encrypted_api_key IS NOT NULL as has_credentials
            FROM users WHERE id = %s
        """, (user_id,))
        row = cur.fetchone()
    return {"has_credentials": row['has_credentials'] if row else False}

@app.get("/api/settings")
async def get_settings(user = Depends(get_current_user)):
    user_id = user['user_id']
    settings = get_user_settings(user_id)
    bot_running = user_id in user_bots and user_bots[user_id].running
    return {
        **settings,
        "available_coins": AVAILABLE_COINS,
        "bot_running": bot_running
    }

@app.put("/api/settings")
async def update_settings(settings: TradingSettings, user = Depends(get_current_user)):
    user_id = user['user_id']

    if user_id in user_bots and user_bots[user_id].running:
        raise HTTPException(status_code=400, detail="Cannot change settings while bot is running. Please stop the bot first.")

    if settings.leverage is not None and (settings.leverage < 1 or settings.leverage > 100):
        raise HTTPException(status_code=400, detail="Leverage must be between 1 and 100")

    if settings.starting_balance is not None and settings.starting_balance < 100:
        raise HTTPException(status_code=400, detail="Starting balance must be at least 100")

    if settings.selected_coins is not None:
        if len(settings.selected_coins) == 0:
            raise HTTPException(status_code=400, detail="Select at least one coin")
        if len(settings.selected_coins) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 coins allowed")
        for coin in settings.selected_coins:
            if coin not in AVAILABLE_COINS:
                raise HTTPException(status_code=400, detail=f"Invalid coin: {coin}")

    if settings.risk_per_trade is not None and (settings.risk_per_trade < 0.001 or settings.risk_per_trade > 0.1):
        raise HTTPException(status_code=400, detail="Risk per trade must be between 0.1% and 10%")

    if settings.stop_loss_pct is not None and (settings.stop_loss_pct < 0.001 or settings.stop_loss_pct > 0.1):
        raise HTTPException(status_code=400, detail="Stop loss must be between 0.1% and 10%")

    if settings.take_profit_pct is not None and (settings.take_profit_pct < 0.001 or settings.take_profit_pct > 0.2):
        raise HTTPException(status_code=400, detail="Take profit must be between 0.1% and 20%")

    if settings.trade_cooldown is not None and (settings.trade_cooldown < 60 or settings.trade_cooldown > 3600):
        raise HTTPException(status_code=400, detail="Trade cooldown must be between 60 and 3600 seconds")

    if settings.min_confidence is not None and (settings.min_confidence < 0.5 or settings.min_confidence > 0.95):
        raise HTTPException(status_code=400, detail="Minimum confidence must be between 50% and 95%")

    if settings.timeframe is not None and settings.timeframe not in VALID_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Timeframe must be one of: {', '.join(VALID_TIMEFRAMES)}")

    if settings.trailing_stop_pct is not None and (settings.trailing_stop_pct < 0.001 or settings.trailing_stop_pct > 0.05):
        raise HTTPException(status_code=400, detail="Trailing stop must be between 0.1% and 5%")

    if settings.max_drawdown_pct is not None and (settings.max_drawdown_pct < 0.05 or settings.max_drawdown_pct > 0.5):
        raise HTTPException(status_code=400, detail="Max drawdown must be between 5% and 50%")

    if settings.retrain_every is not None and (settings.retrain_every < 10 or settings.retrain_every > 500):
        raise HTTPException(status_code=400, detail="Retrain interval must be between 10 and 500 cycles")

    if settings.profit_risk_multiplier is not None and (settings.profit_risk_multiplier < 1.0 or settings.profit_risk_multiplier > 3.0):
        raise HTTPException(status_code=400, detail="Profit risk multiplier must be between 1.0 and 3.0")

    updated = update_user_settings(
        user_id,
        starting_balance=settings.starting_balance,
        leverage=settings.leverage,
        selected_coins=settings.selected_coins,
        risk_per_trade=settings.risk_per_trade,
        stop_loss_pct=settings.stop_loss_pct,
        take_profit_pct=settings.take_profit_pct,
        trade_cooldown=settings.trade_cooldown,
        min_confidence=settings.min_confidence,
        timeframe=settings.timeframe,
        trailing_stop_pct=settings.trailing_stop_pct,
        max_drawdown_pct=settings.max_drawdown_pct,
        retrain_every=settings.retrain_every,
        profit_risk_multiplier=settings.profit_risk_multiplier,
    )
    return {"status": "settings updated", **updated}

@app.post("/api/bot/start")
async def start_bot(user = Depends(get_current_user)):
    user_id = user['user_id']

    perms = get_user_permissions(user_id)
    if perms['account_status'] != 'approved':
        raise HTTPException(status_code=403, detail="Account not approved. Please wait for admin approval.")

    if user_id in user_bots and user_bots[user_id].running:
        return {"status": "already running", "simulation_mode": user_bots[user_id].simulation_mode}

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT encrypted_api_key, encrypted_api_secret, encrypted_api_password
            FROM users WHERE id = %s
        """, (user_id,))
        row = cur.fetchone()

    api_key = decrypt_credential(row['encrypted_api_key']) if row and row['encrypted_api_key'] else None
    api_secret = decrypt_credential(row['encrypted_api_secret']) if row and row['encrypted_api_secret'] else None
    api_password = decrypt_credential(row['encrypted_api_password']) if row and row['encrypted_api_password'] else None

    user_settings = get_user_settings(user_id)

    # Restore actual balance from last recorded performance so compounding
    # continues correctly across restarts instead of resetting to starting_balance.
    perf = get_latest_performance(user_id)
    restored_balance = float(perf['balance']) if perf and perf.get('balance') else user_settings['starting_balance']

    def on_trade(uid, symbol, side, trade_type, size, price, pnl, confidence, reason):
        try:
            save_trade(uid, symbol, side, trade_type, size, price, pnl, confidence, reason)
        except Exception as e:
            print(f"Failed to save trade: {e}")

    def on_signal(uid, signal, confidence, price, rsi, macd, adx):
        try:
            save_signal(uid, int(signal), float(confidence), float(price),
                       float(rsi) if rsi else None, float(macd) if macd else None, float(adx) if adx else None)
        except Exception as e:
            print(f"Failed to save signal: {e}")

    def on_performance(uid, balance, total_pnl, total_trades, winning_trades):
        try:
            save_performance(uid, balance, total_pnl, total_trades, winning_trades)
        except Exception as e:
            print(f"Failed to save performance: {e}")

    bot = TradingService(
        user_id=user_id,
        api_key=api_key,
        api_secret=api_secret,
        api_password=api_password,
        starting_balance=user_settings['starting_balance'],
        leverage=user_settings['leverage'],
        selected_coins=user_settings['selected_coins'],
        risk_per_trade=user_settings['risk_per_trade'],
        stop_loss_pct=user_settings['stop_loss_pct'],
        take_profit_pct=user_settings['take_profit_pct'],
        trade_cooldown=user_settings['trade_cooldown'],
        min_confidence=user_settings['min_confidence'],
        timeframe=user_settings.get('timeframe', '5m'),
        trailing_stop_pct=user_settings.get('trailing_stop_pct', 0.01),
        max_drawdown_pct=user_settings.get('max_drawdown_pct', 0.20),
        retrain_every=user_settings.get('retrain_every', 50),
        profit_risk_multiplier=user_settings.get('profit_risk_multiplier', 1.5),
        on_trade=on_trade,
        on_signal=on_signal,
        on_performance=on_performance
    )
    bot.running = True
    # Restore balance from DB — preserves compound growth across restarts
    bot.balance = restored_balance
    user_bots[user_id] = bot

    asyncio.create_task(run_bot_loop(user_id))

    return {"status": "started", "simulation_mode": bot.simulation_mode, "restored_balance": restored_balance}

@app.post("/api/bot/stop")
async def stop_bot(user = Depends(get_current_user)):
    user_id = user['user_id']
    if user_id in user_bots:
        user_bots[user_id].running = False
        del user_bots[user_id]
    return {"status": "stopped"}

@app.get("/api/bot/status")
async def bot_status(user = Depends(get_current_user)):
    user_id = user['user_id']
    if user_id not in user_bots:
        perf = get_latest_performance(user_id)
        if perf:
            return {
                "running": False,
                "balance": float(perf['balance']),
                "total_pnl": float(perf['total_pnl']),
                "total_trades": perf['total_trades'],
                "winning_trades": perf['winning_trades'],
                "win_rate": (perf['winning_trades'] / perf['total_trades'] * 100) if perf['total_trades'] > 0 else 0
            }
        return {"running": False, "message": "Bot not started"}
    return user_bots[user_id].get_status()

@app.get("/api/trades")
async def get_trades(user = Depends(get_current_user)):
    user_id = user['user_id']
    trades = get_user_trades(user_id, limit=50)
    return {"trades": [dict(t) for t in trades] if trades else []}

@app.get("/api/signals")
async def get_signals(user = Depends(get_current_user)):
    user_id = user['user_id']
    signals = get_user_signals(user_id, limit=20)
    return {"signals": [dict(s) for s in signals] if signals else []}

@app.get("/api/strategies")
async def get_strategies():
    return {
        "name": "Enhanced ML Breakout Detection",
        "description": "SVM model analyzes 30 features to detect volume-confirmed breakouts and breakdowns",
        "components": [
            {
                "name": "Traditional Momentum",
                "weight": "16 features",
                "description": "Core technical indicators for price momentum analysis",
                "indicators": [
                    {"name": "RSI/Stochastic", "desc": "Momentum oscillators for overbought/oversold"},
                    {"name": "MACD/CCI/MFI", "desc": "Trend momentum and money flow"},
                    {"name": "Bollinger/ATR", "desc": "Volatility and price range analysis"},
                    {"name": "ADX", "desc": "Trend strength confirmation (>25)"}
                ]
            },
            {
                "name": "Volume-Price Divergence",
                "weight": "KEY",
                "description": "Detects when volume confirms or diverges from price moves",
                "details": [
                    "Price UP + Volume UP = Strong breakout (BUY)",
                    "Price UP + Volume DOWN = Weak move (AVOID)",
                    "Price DOWN + Volume UP = Strong breakdown (SELL)",
                    "Price DOWN + Volume DOWN = Reversal possible (WATCH)"
                ]
            },
            {
                "name": "VWAP & Institutional",
                "weight": "2 features",
                "description": "Track institutional buying/selling activity",
                "indicators": [
                    {"name": "VWAP Distance", "desc": "Position vs institutional average price"},
                    {"name": "VWAP Slope", "desc": "Institutional momentum direction"}
                ]
            },
            {
                "name": "Breakout Detection",
                "weight": "2 features",
                "description": "Identify breakouts from consolidation patterns",
                "indicators": [
                    {"name": "Breakout Proximity", "desc": "Distance to 20-period high/low"},
                    {"name": "Breakout Quality", "desc": "Volume-confirmed new highs/lows"}
                ]
            },
            {
                "name": "Accumulation/Distribution",
                "weight": "3 features",
                "description": "Smart money flow indicators",
                "indicators": [
                    {"name": "OBV Slope", "desc": "On-Balance Volume trend"},
                    {"name": "AD Slope", "desc": "Accumulation/Distribution line"},
                    {"name": "CMF", "desc": "Chaikin Money Flow (20-period)"}
                ]
            },
            {
                "name": "Squeeze & Vol-Weighted",
                "weight": "5 features",
                "description": "Pre-breakout compression and volume-adjusted signals",
                "indicators": [
                    {"name": "BB Squeeze", "desc": "Bollinger inside Keltner = breakout imminent"},
                    {"name": "Vol-Weighted Mom", "desc": "Price moves weighted by conviction"},
                    {"name": "Directional Volume", "desc": "Net buying vs selling pressure"}
                ]
            }
        ],
        "risk_management": {
            "leverage": "User configurable (1-100x)",
            "risk_per_trade": "0.1% - 10% of balance",
            "stop_loss": "0.1% - 10%",
            "take_profit": "0.1% - 20%",
            "trailing_stop": "0.1% - 5%",
            "max_drawdown": "5% - 50%",
            "trade_cooldown": "1-60 minutes",
            "min_confidence": "50% - 95%"
        }
    }

@app.get("/api/backtest")
async def run_backtest(user = Depends(get_current_user)):
    user_id = user['user_id']
    user_settings = get_user_settings(user_id)

    # Fetch credentials so the backtest uses real exchange data rather than
    # falling back to the BTC-price random-walk simulator.
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT encrypted_api_key, encrypted_api_secret, encrypted_api_password "
                "FROM users WHERE id = %s", (user_id,)
            )
            cred_row = cur.fetchone()
    bt_api_key      = decrypt_credential(cred_row['encrypted_api_key'])      if cred_row and cred_row['encrypted_api_key']      else None
    bt_api_secret   = decrypt_credential(cred_row['encrypted_api_secret'])   if cred_row and cred_row['encrypted_api_secret']   else None
    bt_api_password = decrypt_credential(cred_row['encrypted_api_password']) if cred_row and cred_row['encrypted_api_password'] else None

    bot = TradingService(
        user_id=user_id,
        api_key=bt_api_key,
        api_secret=bt_api_secret,
        api_password=bt_api_password,
        starting_balance=user_settings['starting_balance'],
        leverage=user_settings['leverage'],
        selected_coins=user_settings['selected_coins'],
        risk_per_trade=user_settings['risk_per_trade'],
        stop_loss_pct=user_settings['stop_loss_pct'],
        take_profit_pct=user_settings['take_profit_pct'],
        trade_cooldown=user_settings['trade_cooldown'],
        min_confidence=user_settings['min_confidence'],
        timeframe=user_settings.get('timeframe', '5m'),
        trailing_stop_pct=user_settings.get('trailing_stop_pct', 0.01),
        max_drawdown_pct=user_settings.get('max_drawdown_pct', 0.20),
        retrain_every=user_settings.get('retrain_every', 50),
        profit_risk_multiplier=user_settings.get('profit_risk_multiplier', 1.5),
    )

    # FIX: run_backtest involves ML training (30-120 seconds). Running it directly
    # in an async endpoint blocks the entire event loop — no other requests are
    # served until it finishes. Use run_in_executor to offload to a thread.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, bot.run_backtest, 60)
    return result

def run_optimization_thread(user_id: int, selected_coins: list, starting_balance: float,
                            api_key: str = None, api_secret: str = None, api_password: str = None):
    """Background thread to run optimization"""
    import logging
    import traceback
    import time
    logger = logging.getLogger(__name__)

    last_update = [0]  # Use list for mutable closure

    def progress_callback(progress):
        if progress - last_update[0] >= 5:
            try:
                update_optimization_job(user_id, status='running', progress=progress)
                last_update[0] = progress
                logger.info(f"User {user_id}: Progress {progress:.0f}%")
            except Exception as e:
                logger.warning(f"Failed to update progress: {e}")

    try:
        print(f"[OPTIMIZE] User {user_id}: Starting optimization thread", flush=True)
        print(f"[OPTIMIZE] User {user_id}: API key present: {bool(api_key)}", flush=True)
        print(f"[OPTIMIZE] User {user_id}: Coins: {selected_coins}", flush=True)
        update_optimization_job(user_id, status='running', progress=0)

        optimizer = ParameterOptimizer(
            user_id=user_id,
            selected_coins=selected_coins,
            starting_balance=starting_balance,
            api_key=api_key,
            api_secret=api_secret,
            api_password=api_password
        )

        print(f"[OPTIMIZE] User {user_id}: Starting optimize()", flush=True)
        result = optimizer.optimize(days=60, progress_callback=progress_callback)
        print(f"[OPTIMIZE] User {user_id}: Optimize() complete", flush=True)
        result_json = json.dumps(result)
        update_optimization_job(user_id, status='completed', progress=100, result=result_json)
        save_optimization_run(
            user_id=user_id,
            coins=selected_coins,
            days=60,
            total_tested=result.get('total_tested', 0),
            valid_configs=result.get('valid_configs', 0),
            result=result_json
        )
        time.sleep(0.5)
        print(f"[OPTIMIZE] User {user_id}: Saved to database and history", flush=True)
    except MemoryError as e:
        print(f"[OPTIMIZE] User {user_id}: MEMORY ERROR", flush=True)
        try:
            update_optimization_job(user_id, status='failed', error='Out of memory - try fewer coins')
        except:
            pass
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[OPTIMIZE] User {user_id}: ERROR - {error_msg}", flush=True)
        print(traceback.format_exc(), flush=True)
        try:
            update_optimization_job(user_id, status='failed', error=error_msg)
        except:
            pass

@app.post("/api/optimize/start")
async def start_optimization(user = Depends(get_current_user)):
    """Start parameter optimization in background. Returns immediately."""
    user_id = user['user_id']
    print(f"[START] User {user_id}: Received optimize/start request", flush=True)

    perms = get_user_permissions(user_id)
    if perms['account_status'] != 'approved':
        raise HTTPException(status_code=403, detail="Account not approved. Please wait for admin approval.")
    if not perms['can_optimize']:
        raise HTTPException(status_code=403, detail="Optimization feature not enabled for this account.")

    started = try_start_optimization_job(user_id)
    if not started:
        print(f"[START] User {user_id}: Job already running", flush=True)
        return {"status": "already_running", "message": "Optimization already in progress"}

    print(f"[START] User {user_id}: Job created in DB", flush=True)
    user_settings = get_user_settings(user_id)
    print(f"[START] User {user_id}: Got settings, coins: {user_settings['selected_coins']}", flush=True)

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT encrypted_api_key, encrypted_api_secret, encrypted_api_password FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            print(f"[START] User {user_id}: DB row type: {type(row)}, row: {row}", flush=True)

        if row:
            enc_key = row.get('encrypted_api_key') if hasattr(row, 'get') else row[0]
            enc_secret = row.get('encrypted_api_secret') if hasattr(row, 'get') else row[1]
            enc_pass = row.get('encrypted_api_password') if hasattr(row, 'get') else row[2]
            print(f"[START] User {user_id}: enc_key present: {bool(enc_key)}", flush=True)
            api_key = decrypt_credential(enc_key) if enc_key else None
            api_secret = decrypt_credential(enc_secret) if enc_secret else None
            api_password = decrypt_credential(enc_pass) if enc_pass else None
        else:
            api_key = api_secret = api_password = None
        print(f"[START] User {user_id}: API key present: {bool(api_key)}", flush=True)
    except Exception as e:
        print(f"[START] User {user_id}: ERROR getting credentials: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        api_key = api_secret = api_password = None

    print(f"[START] User {user_id}: Creating thread...", flush=True)
    thread = threading.Thread(
        target=run_optimization_thread,
        args=(user_id, user_settings['selected_coins'], user_settings['starting_balance'],
              api_key, api_secret, api_password)
    )
    thread.daemon = False  # Non-daemon so it keeps running
    thread.start()
    print(f"[START] User {user_id}: Thread started, returning response", flush=True)

    return {"status": "started", "message": "Optimization started. Poll /api/optimize/status for results."}

@app.post("/api/optimize/reset")
async def reset_optimization(user = Depends(get_current_user)):
    """Reset a stuck optimization job."""
    user_id = user['user_id']
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE optimization_jobs SET status = 'failed', error = 'Reset by user' WHERE user_id = %s AND status = 'running'",
                (user_id,)
            )
            conn.commit()
            updated = cur.rowcount
        print(f"[RESET] User {user_id}: Reset {updated} stuck jobs", flush=True)
        return {"status": "ok", "message": f"Reset {updated} stuck job(s)"}
    except Exception as e:
        print(f"[RESET] User {user_id}: ERROR - {e}", flush=True)
        return {"status": "error", "message": str(e)}

@app.get("/api/optimize/status")
async def get_optimization_status(user = Depends(get_current_user)):
    """Poll for optimization status and results."""
    user_id = user['user_id']

    job = get_optimization_job(user_id)

    if not job:
        return {"status": "not_started", "message": "No optimization job found. Start one with POST /api/optimize/start"}

    if job['status'] == 'completed':
        return {
            "status": "completed",
            "result": job['result']
        }
    elif job['status'] == 'failed':
        return {
            "status": "failed",
            "error": job['error']
        }
    else:
        progress = job.get('progress', 0)
        return {
            "status": job['status'],
            "progress": progress,
            "message": f"Optimizing... {progress:.0f}%"
        }

@app.get("/api/optimize")
async def run_optimization(user = Depends(get_current_user)):
    """Legacy endpoint - redirects to polling-based approach"""
    return {"error": "Use POST /api/optimize/start then poll GET /api/optimize/status"}

@app.get("/api/optimize/history")
async def get_optimization_history(user = Depends(get_current_user)):
    """Get list of previous optimization runs."""
    user_id = user['user_id']
    runs = get_optimization_runs(user_id, limit=10)
    return {
        "runs": [{
            "id": r['id'],
            "coins": r['coins'].split(',') if r['coins'] else [],
            "days_tested": r['days_tested'],
            "total_tested": r['total_tested'],
            "valid_configs": r['valid_configs'],
            "completed_at": r['completed_at'].isoformat() if r['completed_at'] else None
        } for r in runs]
    }

@app.get("/api/optimize/run/{run_id}")
async def get_optimization_run_detail(run_id: int, user = Depends(get_current_user)):
    """Get details of a specific optimization run."""
    user_id = user['user_id']
    run = get_optimization_run(run_id, user_id)
    if not run:
        raise HTTPException(status_code=404, detail="Optimization run not found")
    return run

@app.get("/api/optimize/latest")
async def get_latest_optimization(user = Depends(get_current_user)):
    """Get the most recent completed optimization run."""
    user_id = user['user_id']
    run = get_latest_optimization_run(user_id)
    if not run:
        return {"status": "none", "message": "No previous optimization found"}
    return {"status": "found", "run": run}

@app.get("/api/user/permissions")
async def get_my_permissions(user = Depends(get_current_user)):
    user_id = user['user_id']
    perms = get_user_permissions(user_id)
    return perms

@app.get("/api/admin/users")
async def admin_get_all_users(user = Depends(get_current_user)):
    user_id = user['user_id']
    perms = get_user_permissions(user_id)
    if not perms['is_admin']:
        raise HTTPException(status_code=403, detail="Admin access required")
    users = get_all_users()
    return {"users": [dict(u) for u in users]}

@app.get("/api/admin/users/pending")
async def admin_get_pending_users(user = Depends(get_current_user)):
    user_id = user['user_id']
    perms = get_user_permissions(user_id)
    if not perms['is_admin']:
        raise HTTPException(status_code=403, detail="Admin access required")
    users = get_pending_users()
    return {"users": [dict(u) for u in users]}

class UserApproval(BaseModel):
    action: str  # approve, reject
    can_optimize: Optional[bool] = True
    can_live_trade: Optional[bool] = False

@app.post("/api/admin/users/{target_user_id}/approve")
async def admin_approve_user(target_user_id: int, approval: UserApproval, user = Depends(get_current_user)):
    user_id = user['user_id']
    perms = get_user_permissions(user_id)
    if not perms['is_admin']:
        raise HTTPException(status_code=403, detail="Admin access required")

    if approval.action == 'approve':
        update_user_status(target_user_id, 'approved')
        update_user_permissions(target_user_id,
                               can_optimize=approval.can_optimize,
                               can_live_trade=approval.can_live_trade)
        return {"status": "approved", "user_id": target_user_id}
    elif approval.action == 'reject':
        update_user_status(target_user_id, 'rejected')
        return {"status": "rejected", "user_id": target_user_id}
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

class PermissionUpdate(BaseModel):
    can_optimize: Optional[bool] = None
    can_live_trade: Optional[bool] = None
    is_admin: Optional[bool] = None

@app.post("/api/admin/users/{target_user_id}/permissions")
async def admin_update_permissions(target_user_id: int, perms_update: PermissionUpdate, user = Depends(get_current_user)):
    user_id = user['user_id']
    perms = get_user_permissions(user_id)
    if not perms['is_admin']:
        raise HTTPException(status_code=403, detail="Admin access required")

    update_user_permissions(target_user_id,
                           can_optimize=perms_update.can_optimize,
                           can_live_trade=perms_update.can_live_trade,
                           is_admin=perms_update.is_admin)
    return {"status": "updated", "user_id": target_user_id}

async def run_bot_loop(user_id: int):
    while user_id in user_bots and user_bots[user_id].running:
        try:
            bot = user_bots[user_id]
            signal_data = bot.run_cycle()

            if signal_data and user_id in user_connections:
                status = bot.get_status()
                for ws in user_connections[user_id]:
                    try:
                        await ws.send_json({
                            "type": "update",
                            "signal": signal_data,
                            "status": status
                        })
                    except:
                        pass

        except Exception as e:
            print(f"Bot loop error for user {user_id}: {e}")

        # FIX: align sleep with the bot's actual timeframe instead of hardcoded 30s.
        # On a 1h chart, sleeping 30s means 120 wasted API calls per candle.
        # Sleeping 85% of the candle duration fires just before each close.
        tf = user_bots[user_id].timeframe if user_id in user_bots else '5m'
        await asyncio.sleep(_bot_sleep_seconds(tf))

    if user_id in user_bots:
        del user_bots[user_id]

@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    payload = decode_token(token)
    if not payload:
        await websocket.close(code=4001)
        return

    user_id = payload['user_id']
    await websocket.accept()

    if user_id not in user_connections:
        user_connections[user_id] = []
    user_connections[user_id].append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            if data == "status" and user_id in user_bots:
                await websocket.send_json(user_bots[user_id].get_status())
    except WebSocketDisconnect:
        if user_id in user_connections and websocket in user_connections[user_id]:
            user_connections[user_id].remove(websocket)
            if not user_connections[user_id]:
                del user_connections[user_id]

static_dir = Path(__file__).parent.parent / "static" / "dist"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    if full_path.startswith("api/") or full_path.startswith("ws"):
        raise HTTPException(status_code=404)

    static_dir = Path(__file__).parent.parent / "static" / "dist"
    if static_dir.exists():
        file_path = static_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(static_dir / "index.html")

    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Trading Bot</title>
    <style>
        body { font-family: system-ui; background: #1a1a2e; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .container { text-align: center; padding: 20px; }
        h1 { color: #e94560; }
        p { color: #a0a0a0; }
        code { background: #0f3460; padding: 4px 8px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Crypto Trading Bot API</h1>
        <p>Backend is running. Build the mobile app with:</p>
        <p><code>cd mobile && npm run build</code></p>
    </div>
</body>
</html>
    """)
