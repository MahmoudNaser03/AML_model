"""Bounded, expiring SQLite explanation cache shared by local Serve replicas.

Stores engineered values and decisions only; no raw account IDs or transaction rows.
For a multinode deployment replace this with a shared network cache.
"""
import json
import sqlite3
import time
import zlib
from pathlib import Path

import pandas as pd


class BatchCache:
    def __init__(self, path, ttl=900, max_bytes=64 * 1024 * 1024, max_batches=10):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl, self.max_bytes, self.max_batches = ttl, max_bytes, max_batches
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, created REAL NOT NULL, version TEXT NOT NULL, payload BLOB NOT NULL)")

    def _connection(self):
        return sqlite3.connect(self.path, timeout=10)

    def put(self, batch_id, X, decisions, version):
        ids = decisions.index[decisions.alert].to_numpy()
        payload = zlib.compress(json.dumps({"features": X.loc[ids].to_dict("split"),
                    "decisions": decisions.loc[ids].to_dict("split")}, allow_nan=False).encode())
        if len(payload) > self.max_bytes:
            return False
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM batches WHERE created < ?", (time.time() - self.ttl,))
            db.execute("INSERT INTO batches VALUES (?, ?, ?, ?)", (batch_id, time.time(), version, payload))
            while True:
                count, size = db.execute("SELECT count(*), coalesce(sum(length(payload)), 0) FROM batches").fetchone()
                if count <= self.max_batches and size <= self.max_bytes: break
                db.execute("DELETE FROM batches WHERE id=(SELECT id FROM batches ORDER BY created LIMIT 1)")
        return True

    def get(self, batch_id, version):
        with self._connection() as db:
            row = db.execute("SELECT payload FROM batches WHERE id=? AND version=? AND created>=?", (batch_id, version, time.time() - self.ttl)).fetchone()
        if row is None: return None
        payload = json.loads(zlib.decompress(row[0]))
        return pd.DataFrame(**payload["features"]), pd.DataFrame(**payload["decisions"])
