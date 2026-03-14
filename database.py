"""
database.py
===========
SQLite database layer for the XAI Bankruptcy SaaS Platform.
Tables: users, subscriptions, predictions, user_contributions
"""

import sqlite3
import json
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "instance/app.db"

# Ensure the instance directory exists
import os as _os
_os.makedirs(_os.path.dirname(DB_PATH), exist_ok=True)


# ── Connection ────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            is_admin      INTEGER DEFAULT 0,
            created_at    TEXT    DEFAULT (datetime('now')),
            last_login    TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL REFERENCES users(id),
            company_name  TEXT    DEFAULT 'Unknown Company',
            inputs        TEXT    NOT NULL,
            results       TEXT    NOT NULL,
            valuation_models TEXT,
            xai_data      TEXT,
            risk_level    TEXT,
            probability   REAL,
            created_at    TEXT    DEFAULT (datetime('now'))
        );
        """)
        conn.commit()
    print("[DB] Database initialised.")


# ── User CRUD ─────────────────────────────────────────────────────────────────

def create_user(name: str, email: str, password: str) -> int:
    h = generate_password_hash(password)
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?,?,?)",
            (name, email, h)
        )
        user_id = cur.lastrowid
        conn.commit()
    return user_id


def get_user_by_email(email: str):
    with get_db() as conn:
        return conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()


def get_user_by_id(user_id: int):
    with get_db() as conn:
        return conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()


def verify_password(email: str, password: str):
    user = get_user_by_email(email)
    if user and check_password_hash(user["password_hash"], password):
        with get_db() as conn:
            conn.execute("UPDATE users SET last_login=datetime('now') WHERE id=?", (user["id"],))
            conn.commit()
        return user
    return None


def update_user(user_id: int, name: str = None, email: str = None):
    with get_db() as conn:
        if name:
            conn.execute("UPDATE users SET name=? WHERE id=?", (name, user_id))
        if email:
            conn.execute("UPDATE users SET email=? WHERE id=?", (email, user_id))
        conn.commit()




# ── Predictions ───────────────────────────────────────────────────────────────

def save_prediction(user_id: int, company_name: str, inputs: dict,
                    results: dict, valuation: dict, xai_data: dict,
                    risk_level: str, probability: float) -> int:
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO predictions
            (user_id, company_name, inputs, results, valuation_models, xai_data, risk_level, probability)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            user_id, company_name,
            json.dumps(inputs), json.dumps(results),
            json.dumps(valuation), json.dumps(xai_data),
            risk_level, probability
        ))
        conn.commit()
        return cur.lastrowid


def get_user_predictions(user_id: int, limit: int = 50):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions WHERE user_id=?
            ORDER BY created_at DESC LIMIT ?
        """, (user_id, limit)).fetchall()
    return rows


def get_prediction_by_id(pred_id: int, user_id: int):
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM predictions WHERE id=? AND user_id=?",
            (pred_id, user_id)
        ).fetchone()





# ── Admin Stats ───────────────────────────────────────────────────────────────

def get_admin_stats():
    with get_db() as conn:
        stats = {
            "total_users":       conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            "total_predictions": conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
        }
    return stats




# ── User stats ────────────────────────────────────────────────────────────────

def get_user_stats(user_id: int) -> dict:
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE user_id=?", (user_id,)
        ).fetchone()[0]
        high = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE user_id=? AND risk_level='HIGH'", (user_id,)
        ).fetchone()[0]
    return {"total_predictions": total, "high_risk": high}