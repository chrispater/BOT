import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from datetime import datetime
from dotenv import load_dotenv

# Load .env file from multiple possible locations
load_dotenv()  # Current directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))  # Parent directory

DATABASE_URL = os.getenv('DATABASE_URL')

def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                encrypted_api_key TEXT,
                encrypted_api_secret TEXT,
                encrypted_api_password TEXT,
                starting_balance DECIMAL(18, 8) DEFAULT 10000,
                leverage INTEGER DEFAULT 10,
                selected_coins TEXT DEFAULT 'BTC/USDT:USDT',
                risk_per_trade DECIMAL(5, 4) DEFAULT 0.02,
                stop_loss_pct DECIMAL(5, 4) DEFAULT 0.015,
                take_profit_pct DECIMAL(5, 4) DEFAULT 0.03,
                trade_cooldown INTEGER DEFAULT 300,
                min_confidence DECIMAL(5, 4) DEFAULT 0.65,
                timeframe VARCHAR(10) DEFAULT '5m',
                trailing_stop_pct DECIMAL(5, 4) DEFAULT 0.01,
                max_drawdown_pct DECIMAL(5, 4) DEFAULT 0.20,
                retrain_every INTEGER DEFAULT 50,
                profit_risk_multiplier DECIMAL(5, 4) DEFAULT 1.5,
                is_admin BOOLEAN DEFAULT FALSE,
                account_status VARCHAR(20) DEFAULT 'pending',
                can_optimize BOOLEAN DEFAULT FALSE,
                can_live_trade BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # ADD IF NOT EXISTS handles both fresh installs and existing databases
        columns_to_add = [
            ('starting_balance', 'DECIMAL(18, 8)', '10000'),
            ('leverage', 'INTEGER', '10'),
            ('selected_coins', 'TEXT', "'BTC/USDT:USDT'"),
            ('risk_per_trade', 'DECIMAL(5, 4)', '0.02'),
            ('stop_loss_pct', 'DECIMAL(5, 4)', '0.015'),
            ('take_profit_pct', 'DECIMAL(5, 4)', '0.03'),
            ('trade_cooldown', 'INTEGER', '300'),
            ('min_confidence', 'DECIMAL(5, 4)', '0.65'),
            ('timeframe', 'VARCHAR(10)', "'5m'"),
            ('trailing_stop_pct', 'DECIMAL(5, 4)', '0.01'),
            ('max_drawdown_pct', 'DECIMAL(5, 4)', '0.20'),
            ('retrain_every', 'INTEGER', '50'),
            ('profit_risk_multiplier', 'DECIMAL(5, 4)', '1.5'),
            ('is_admin', 'BOOLEAN', 'FALSE'),
            ('account_status', 'VARCHAR(20)', "'pending'"),
            ('can_optimize', 'BOOLEAN', 'FALSE'),
            ('can_live_trade', 'BOOLEAN', 'FALSE'),
            ('simulation_mode', 'BOOLEAN', 'TRUE'),
            ('adx_threshold', 'INTEGER', '18'),
            ('reliability_gate', 'BOOLEAN', 'TRUE'),
            ('reliability_min_winrate', 'DECIMAL(5, 4)', '0.60'),
        ]
        for col, col_type, default in columns_to_add:
            try:
                cur.execute(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col} {col_type} DEFAULT {default}")
            except:
                pass
        cur.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                symbol VARCHAR(50) NOT NULL,
                side VARCHAR(10) NOT NULL,
                trade_type VARCHAR(20) NOT NULL,
                size DECIMAL(18, 8) NOT NULL,
                price DECIMAL(18, 8) NOT NULL,
                pnl DECIMAL(18, 8),
                confidence DECIMAL(5, 4),
                reason VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                balance DECIMAL(18, 8) NOT NULL,
                total_pnl DECIMAL(18, 8) DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                signal INTEGER NOT NULL,
                confidence DECIMAL(5, 4) NOT NULL,
                price DECIMAL(18, 8) NOT NULL,
                rsi DECIMAL(10, 4),
                macd DECIMAL(18, 8),
                adx DECIMAL(10, 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS optimization_jobs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) UNIQUE,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                result TEXT,
                error TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                coins TEXT,
                days_tested INTEGER DEFAULT 30,
                total_tested INTEGER DEFAULT 0,
                valid_configs INTEGER DEFAULT 0,
                result TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        for col, typ in [('best_roi', 'REAL'), ('best_monthly_roi', 'REAL'), ('best_win_rate', 'REAL')]:
            try:
                cur.execute(f'ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS {col} {typ}')
            except Exception:
                pass
        conn.commit()

def save_trade(user_id: int, symbol: str, side: str, trade_type: str, size: float,
               price: float, pnl: float = None, confidence: float = None, reason: str = None):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO trades (user_id, symbol, side, trade_type, size, price, pnl, confidence, reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (user_id, symbol, side, trade_type, size, price, pnl, confidence, reason))
        return cur.fetchone()['id']

def save_signal(user_id: int, signal: int, confidence: float, price: float,
                rsi: float = None, macd: float = None, adx: float = None):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO signals (user_id, signal, confidence, price, rsi, macd, adx)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (user_id, signal, confidence, price, rsi, macd, adx))

def save_performance(user_id: int, balance: float, total_pnl: float,
                     total_trades: int, winning_trades: int):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO performance (user_id, balance, total_pnl, total_trades, winning_trades)
            VALUES (%s, %s, %s, %s, %s)
        ''', (user_id, balance, total_pnl, total_trades, winning_trades))

def get_user_trades(user_id: int, limit: int = 50):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT * FROM trades WHERE user_id = %s
            ORDER BY created_at DESC LIMIT %s
        ''', (user_id, limit))
        return cur.fetchall()

def get_user_signals(user_id: int, limit: int = 20):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT * FROM signals WHERE user_id = %s
            ORDER BY created_at DESC LIMIT %s
        ''', (user_id, limit))
        return cur.fetchall()

def get_latest_performance(user_id: int):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT * FROM performance WHERE user_id = %s
            ORDER BY recorded_at DESC LIMIT 1
        ''', (user_id,))
        return cur.fetchone()

def get_user_settings(user_id: int):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT starting_balance, leverage, selected_coins,
                   risk_per_trade, stop_loss_pct, take_profit_pct,
                   trade_cooldown, min_confidence, timeframe,
                   trailing_stop_pct, max_drawdown_pct, retrain_every,
                   profit_risk_multiplier, simulation_mode, adx_threshold,
                   reliability_gate, reliability_min_winrate
            FROM users WHERE id = %s
        ''', (user_id,))
        row = cur.fetchone()
        if row:
            return {
                'starting_balance': float(row['starting_balance']) if row['starting_balance'] else 10000,
                'leverage': int(row['leverage']) if row['leverage'] else 10,
                'selected_coins': row['selected_coins'].split(',') if row['selected_coins'] else ['BTC/USDT:USDT'],
                'risk_per_trade': float(row['risk_per_trade']) if row['risk_per_trade'] else 0.02,
                'stop_loss_pct': float(row['stop_loss_pct']) if row['stop_loss_pct'] else 0.015,
                'take_profit_pct': float(row['take_profit_pct']) if row['take_profit_pct'] else 0.03,
                'trade_cooldown': int(row['trade_cooldown']) if row['trade_cooldown'] else 300,
                'min_confidence': float(row['min_confidence']) if row['min_confidence'] else 0.65,
                'timeframe': row['timeframe'] if row.get('timeframe') else '5m',
                'trailing_stop_pct': float(row['trailing_stop_pct']) if row.get('trailing_stop_pct') else 0.01,
                'max_drawdown_pct': float(row['max_drawdown_pct']) if row.get('max_drawdown_pct') else 0.20,
                'retrain_every': int(row['retrain_every']) if row.get('retrain_every') else 50,
                'profit_risk_multiplier': float(row['profit_risk_multiplier']) if row.get('profit_risk_multiplier') else 1.5,
                'simulation_mode': bool(row['simulation_mode']) if row.get('simulation_mode') is not None else True,
                'adx_threshold': int(row['adx_threshold']) if row.get('adx_threshold') is not None else 18,
                'reliability_gate': bool(row['reliability_gate']) if row.get('reliability_gate') is not None else True,
                'reliability_min_winrate': float(row['reliability_min_winrate']) if row.get('reliability_min_winrate') is not None else 0.60,
            }
        return {
            'starting_balance': 10000,
            'leverage': 10,
            'selected_coins': ['BTC/USDT:USDT'],
            'risk_per_trade': 0.02,
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03,
            'trade_cooldown': 300,
            'min_confidence': 0.65,
            'timeframe': '5m',
            'trailing_stop_pct': 0.01,
            'max_drawdown_pct': 0.20,
            'retrain_every': 50,
            'profit_risk_multiplier': 1.5,
            'simulation_mode': True,
            'adx_threshold': 18,
            'reliability_gate': True,
            'reliability_min_winrate': 0.60,
        }

def update_user_settings(user_id: int, starting_balance: float = None,
                         leverage: int = None, selected_coins: list = None,
                         risk_per_trade: float = None, stop_loss_pct: float = None,
                         take_profit_pct: float = None, trade_cooldown: int = None,
                         min_confidence: float = None, timeframe: str = None,
                         trailing_stop_pct: float = None, max_drawdown_pct: float = None,
                         retrain_every: int = None, profit_risk_multiplier: float = None,
                         simulation_mode: bool = None, adx_threshold: int = None,
                         reliability_gate: bool = None, reliability_min_winrate: float = None):
    with get_db() as conn:
        cur = conn.cursor()
        updates = []
        values = []
        if starting_balance is not None:
            updates.append("starting_balance = %s")
            values.append(starting_balance)
        if leverage is not None:
            updates.append("leverage = %s")
            values.append(leverage)
        if selected_coins is not None:
            updates.append("selected_coins = %s")
            values.append(','.join(selected_coins))
        if risk_per_trade is not None:
            updates.append("risk_per_trade = %s")
            values.append(risk_per_trade)
        if stop_loss_pct is not None:
            updates.append("stop_loss_pct = %s")
            values.append(stop_loss_pct)
        if take_profit_pct is not None:
            updates.append("take_profit_pct = %s")
            values.append(take_profit_pct)
        if trade_cooldown is not None:
            updates.append("trade_cooldown = %s")
            values.append(trade_cooldown)
        if min_confidence is not None:
            updates.append("min_confidence = %s")
            values.append(min_confidence)
        if timeframe is not None:
            updates.append("timeframe = %s")
            values.append(timeframe)
        if trailing_stop_pct is not None:
            updates.append("trailing_stop_pct = %s")
            values.append(trailing_stop_pct)
        if max_drawdown_pct is not None:
            updates.append("max_drawdown_pct = %s")
            values.append(max_drawdown_pct)
        if retrain_every is not None:
            updates.append("retrain_every = %s")
            values.append(retrain_every)
        if profit_risk_multiplier is not None:
            updates.append("profit_risk_multiplier = %s")
            values.append(profit_risk_multiplier)
        if simulation_mode is not None:
            updates.append("simulation_mode = %s")
            values.append(simulation_mode)
        if adx_threshold is not None:
            updates.append("adx_threshold = %s")
            values.append(int(adx_threshold))
        if reliability_gate is not None:
            updates.append("reliability_gate = %s")
            values.append(bool(reliability_gate))
        if reliability_min_winrate is not None:
            updates.append("reliability_min_winrate = %s")
            values.append(float(reliability_min_winrate))
        if updates:
            values.append(user_id)
            cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = %s", values)
            conn.commit()
        return get_user_settings(user_id)

def save_optimization_job(user_id: int, status: str, progress: int = 0, result: str = None, error: str = None):
    import json
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO optimization_jobs (user_id, status, progress, result, error, started_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) DO UPDATE SET
                status = EXCLUDED.status,
                progress = EXCLUDED.progress,
                result = EXCLUDED.result,
                error = EXCLUDED.error,
                started_at = CASE WHEN EXCLUDED.status = 'starting' THEN CURRENT_TIMESTAMP ELSE optimization_jobs.started_at END,
                completed_at = CASE WHEN EXCLUDED.status IN ('completed', 'failed') THEN CURRENT_TIMESTAMP ELSE NULL END
        ''', (user_id, status, progress, result, error))
        conn.commit()

def try_start_optimization_job(user_id: int):
    """Atomically try to start an optimization job. Returns True if started, False if already running."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO optimization_jobs (user_id, status, progress, started_at)
            VALUES (%s, 'starting', 0, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) DO UPDATE SET
                status = 'starting',
                progress = 0,
                result = NULL,
                error = NULL,
                started_at = CURRENT_TIMESTAMP,
                completed_at = NULL
            WHERE optimization_jobs.status NOT IN ('starting', 'running')
            RETURNING id
        ''', (user_id,))
        result = cur.fetchone()
        conn.commit()
        return result is not None

def update_optimization_job(user_id: int, status: str = None, progress: int = None, result: str = None, error: str = None):
    import logging
    logger = logging.getLogger(__name__)
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        updates = []
        values = []
        if status is not None:
            updates.append("status = %s")
            values.append(status)
        if progress is not None:
            updates.append("progress = %s")
            values.append(progress)
        if result is not None:
            updates.append("result = %s")
            values.append(result)
        if error is not None:
            updates.append("error = %s")
            values.append(error)
        if status in ('completed', 'failed'):
            updates.append("completed_at = CURRENT_TIMESTAMP")
        if updates:
            values.append(user_id)
            sql = f"UPDATE optimization_jobs SET {', '.join(updates)} WHERE user_id = %s"
            cur.execute(sql, values)
            rows_affected = cur.rowcount
            conn.commit()
            logger.info(f"DB COMMITTED: user_id={user_id}, status={status}, rows={rows_affected}")
    except Exception as e:
        logger.error(f"update_optimization_job FAILED: user_id={user_id}, error={str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def get_optimization_job(user_id: int):
    import json
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT status, progress, result, error, started_at, completed_at
            FROM optimization_jobs WHERE user_id = %s
        ''', (user_id,))
        row = cur.fetchone()
        if row:
            result_data = None
            if row['result']:
                try:
                    result_data = json.loads(row['result'])
                except:
                    result_data = row['result']
            return {
                'status': row['status'],
                'progress': row['progress'],
                'result': result_data,
                'error': row['error'],
                'started_at': row['started_at'].isoformat() if row['started_at'] else None,
                'completed_at': row['completed_at'].isoformat() if row['completed_at'] else None
            }
        return None

def save_optimization_run(user_id: int, coins: list, days: int, total_tested: int, valid_configs: int, result: str,
                          best_roi: float = 0, best_monthly_roi: float = 0, best_win_rate: float = 0):
    import json
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO optimization_runs (user_id, status, coins, days_tested, total_tested, valid_configs, result, best_roi, best_monthly_roi, best_win_rate, completed_at)
            VALUES (%s, 'completed', %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
        ''', (user_id, ','.join(coins), days, total_tested, valid_configs, result, best_roi, best_monthly_roi, best_win_rate))
        return cur.fetchone()['id']

def get_optimization_runs(user_id: int, limit: int = 10):
    import json
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT id, coins, days_tested, total_tested, valid_configs,
                   best_roi, best_monthly_roi, best_win_rate,
                   started_at, completed_at
            FROM optimization_runs WHERE user_id = %s
            ORDER BY completed_at DESC LIMIT %s
        ''', (user_id, limit))
        return cur.fetchall()

def get_optimization_run(run_id: int, user_id: int):
    import json
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT id, coins, days_tested, total_tested, valid_configs, result,
                   best_roi, best_monthly_roi, best_win_rate,
                   started_at, completed_at
            FROM optimization_runs WHERE id = %s AND user_id = %s
        ''', (run_id, user_id))
        row = cur.fetchone()
        if row:
            result_data = None
            if row['result']:
                try:
                    result_data = json.loads(row['result'])
                except:
                    result_data = row['result']
            return {
                'id': row['id'],
                'coins': row['coins'].split(',') if row['coins'] else [],
                'days_tested': row['days_tested'],
                'total_tested': row['total_tested'],
                'valid_configs': row['valid_configs'],
                'result': result_data,
                'best_roi': row['best_roi'],
                'best_monthly_roi': row['best_monthly_roi'],
                'best_win_rate': row['best_win_rate'],
                'started_at': row['started_at'].isoformat() if row['started_at'] else None,
                'completed_at': row['completed_at'].isoformat() if row['completed_at'] else None
            }
        return None

def get_latest_optimization_run(user_id: int):
    import json
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT id, coins, days_tested, total_tested, valid_configs, result,
                   best_roi, best_monthly_roi, best_win_rate,
                   started_at, completed_at
            FROM optimization_runs WHERE user_id = %s
            ORDER BY completed_at DESC LIMIT 1
        ''', (user_id,))
        row = cur.fetchone()
        if row:
            result_data = None
            if row['result']:
                try:
                    result_data = json.loads(row['result'])
                except:
                    result_data = row['result']
            return {
                'id': row['id'],
                'coins': row['coins'].split(',') if row['coins'] else [],
                'days_tested': row['days_tested'],
                'total_tested': row['total_tested'],
                'valid_configs': row['valid_configs'],
                'result': result_data,
                'best_roi': row['best_roi'],
                'best_monthly_roi': row['best_monthly_roi'],
                'best_win_rate': row['best_win_rate'],
                'started_at': row['started_at'].isoformat() if row['started_at'] else None,
                'completed_at': row['completed_at'].isoformat() if row['completed_at'] else None
            }
        return None

def get_user_permissions(user_id: int):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT is_admin, account_status, can_optimize, can_live_trade
            FROM users WHERE id = %s
        ''', (user_id,))
        row = cur.fetchone()
        if row:
            return {
                'is_admin': row['is_admin'] if row['is_admin'] else False,
                'account_status': row['account_status'] if row['account_status'] else 'pending',
                'can_optimize': row['can_optimize'] if row['can_optimize'] else False,
                'can_live_trade': row['can_live_trade'] if row['can_live_trade'] else False
            }
        return {'is_admin': False, 'account_status': 'pending', 'can_optimize': False, 'can_live_trade': False}

def get_pending_users():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT id, username, created_at, account_status, can_optimize, can_live_trade
            FROM users WHERE account_status = 'pending'
            ORDER BY created_at DESC
        ''')
        return cur.fetchall()

def get_all_users():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT id, username, created_at, account_status, is_admin, can_optimize, can_live_trade
            FROM users ORDER BY created_at DESC
        ''')
        return cur.fetchall()

def update_user_status(user_id: int, status: str):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('UPDATE users SET account_status = %s WHERE id = %s', (status, user_id))

def update_user_permissions(user_id: int, can_optimize: bool = None, can_live_trade: bool = None, is_admin: bool = None):
    with get_db() as conn:
        cur = conn.cursor()
        updates = []
        values = []
        if can_optimize is not None:
            updates.append("can_optimize = %s")
            values.append(can_optimize)
        if can_live_trade is not None:
            updates.append("can_live_trade = %s")
            values.append(can_live_trade)
        if is_admin is not None:
            updates.append("is_admin = %s")
            values.append(is_admin)
        if updates:
            values.append(user_id)
            cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = %s", values)

def make_user_admin(user_id: int):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('UPDATE users SET is_admin = TRUE, account_status = %s, can_optimize = TRUE, can_live_trade = TRUE WHERE id = %s', ('approved', user_id))
