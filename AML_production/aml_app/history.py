"""Concurrent, bounded dashboard history; storage is opt-in."""
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd


def save_history(path, rows, batch_id, retention_days=30):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=10) as db:
        db.execute("CREATE TABLE IF NOT EXISTS history (id TEXT PRIMARY KEY, created REAL, payload TEXT)")
        db.execute("DELETE FROM history WHERE created < ?", (time.time() - retention_days * 86400,))
        db.execute("INSERT OR REPLACE INTO history VALUES (?, ?, ?)", (batch_id, time.time(), rows.to_json(orient="records")))


def load_history(path, retention_days=30, max_rows=100000):
    if not Path(path).exists(): return pd.DataFrame()
    with sqlite3.connect(path, timeout=10) as db:
        try:
            records = db.execute("SELECT payload FROM history WHERE created>=? ORDER BY created DESC", (time.time() - retention_days * 86400,))
            rows = []
            for payload, in records:
                rows.extend(json.loads(payload))
                if len(rows) >= max_rows: break
        except sqlite3.OperationalError:
            return pd.DataFrame()
    return pd.DataFrame(rows[:max_rows])
