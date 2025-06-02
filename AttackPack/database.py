import sqlite3
import torch
from dataclasses import dataclass

@dataclass
class AttackEntry:
    model_name: str
    weights_file: str
    epsilon: float
    attack_type: str
    extent: str                 # "data", "rr", or "both"
    data_tensor: torch.Tensor   # adversarial-modified signal
    rr_tensor: torch.Tensor     # adversarial-modified signal

class AttackDatabase:
    def __init__(self, db_path="attacks.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._setup_schema()

    def _setup_schema(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS attacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                weights_file TEXT NOT NULL,
                epsilon REAL NOT NULL,
                attack_type TEXT NOT NULL,
                extent TEXT NOT NULL CHECK (extent IN ('data', 'rr', 'both')),
                data_blob BLOB,
                rr_blob BLOB
            )
        """)
        self.conn.commit()

    def insert(self, entry: AttackEntry):
        data_blob = sqlite3.Binary(entry.data_tensor.numpy().tobytes()) if entry.extent in ('data', 'both') else None
        rr_blob = sqlite3.Binary(entry.rr_tensor.numpy().tobytes()) if entry.extent in ('rr', 'both') else None

        self.cur.execute("""
            INSERT INTO attacks (
                model_name, weights_file, epsilon, attack_type, extent, data_blob, rr_blob
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.model_name,
            entry.weights_file,
            entry.epsilon,
            entry.attack_type,
            entry.extent,
            data_blob,
            rr_blob
        ))
        self.conn.commit()

    def query(self, where_clause="", params=()):
        sql = "SELECT model_name, weights_file, epsilon, attack_type, extent, data_blob, rr_blob FROM attacks"
        if where_clause:
            sql += f" WHERE {where_clause}"
        self.cur.execute(sql, params)
        return self.cur.fetchall()

    def load_tensor(self, blob: bytes, shape, dtype=torch.float32):
        return torch.frombuffer(blob, dtype=dtype).reshape(shape)

    def close(self):
        self.conn.close()


# Create a global instance of the AttackDatabase
DATABASE = AttackDatabase()
