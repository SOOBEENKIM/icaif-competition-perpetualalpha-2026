"""Durable execution state. A claim is committed BEFORE sending an order."""
import json
import sqlite3
import time


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS orders (
                cloid TEXT PRIMARY KEY, slot TEXT UNIQUE NOT NULL, status TEXT NOT NULL,
                plan TEXT NOT NULL, result TEXT, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL, id TEXT NOT NULL, raw TEXT NOT NULL, PRIMARY KEY(kind,id));
        """)

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value, allow_nan=False)))

    def claim(self, slot, order):
        """False means this round/coin/phase was already attempted, even after a crash."""
        with self.db:
            result = self.db.execute("INSERT OR IGNORE INTO orders VALUES (?,?, 'sending', ?,NULL,?)",
                (order["cloid"], slot, json.dumps(order, allow_nan=False), time.time()))
        return result.rowcount == 1

    def update(self, cloid, status, result):
        with self.db:
            self.db.execute("UPDATE orders SET status=?,result=?,updated=? WHERE cloid=?",
                            (status, json.dumps(result, allow_nan=False), time.time(), cloid))

    def orders(self):
        return [{"cloid": c, "slot": s, "status": t, "plan": json.loads(p),
                 "result": json.loads(r) if r else None}
                for c, s, t, p, r in self.db.execute("SELECT cloid,slot,status,plan,result FROM orders ORDER BY updated")]

    def add_records(self, kind, records):
        with self.db:
            for key, row in records:
                self.db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?)",
                                (kind, key, json.dumps(row, sort_keys=True, allow_nan=False)))

    def records(self, kind):
        return [json.loads(r[0]) for r in self.db.execute("SELECT raw FROM records WHERE kind=?", (kind,))]

    def close(self):
        self.db.close()
