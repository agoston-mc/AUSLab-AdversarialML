import os.path
import sqlite3

import numpy as np
import torch
from dataclasses import dataclass

DATA_SHAPE = (1, 300)
RR_SHAPE = (1, 4)

@dataclass
class AttackEntry:
    model_name: str
    weights_file: str
    epsilon: float
    attack_type: str
    extent: str                 # "data", "rr", or "both"
    data_idx: int               # index of the data sample used for the attack
    data_file: str              # file name of the data sample
    data_noise: torch.Tensor    # noise added to the data signal
    rr_noise: torch.Tensor      # noise added to the rr signal

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
                data_idx INTEGER,
                data_file TEXT,
                data_noise BLOB,
                rr_noise BLOB
            )
        """)
        self.conn.commit()

    def insert(self, entry: AttackEntry):
        data_noise = sqlite3.Binary(entry.data_noise.cpu().detach().numpy().tobytes()) if entry.extent in ('data', 'both') else None
        rr_noise = sqlite3.Binary(entry.rr_noise.cpu().detach().numpy().tobytes()) if entry.extent in ('rr', 'both') else None

        self.cur.execute("""
            INSERT INTO attacks (
                model_name, weights_file, epsilon, attack_type, extent, data_idx, data_file, data_noise, rr_noise
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.model_name,
            entry.weights_file,
            entry.epsilon,
            entry.attack_type,
            entry.extent,
            entry.data_idx,
            entry.data_file,
            data_noise,
            rr_noise
        ))
        self.conn.commit()

    def query(self, where_clause="", params=()):
        sql = "SELECT model_name, weights_file, epsilon, attack_type, extent, data_idx, data_file, data_noise, rr_noise FROM attacks"
        if where_clause:
            sql += f" WHERE {where_clause}"
        self.cur.execute(sql, params)
        return self.cur.fetchall()

    def reset_db(self):
        self.cur.execute("DROP TABLE IF EXISTS attacks")

    def __getitem__(self, item):
        if isinstance(item, int):
            where_clause = f"id is {item}"

        if isinstance(item, slice):
            where_clause = f"id >= {item.start} and id < {item.stop}"

        if isinstance(item, tuple):
            where_clause = " or ".join([f"id = {item[0]}"])

        tab = self.query(where_clause)

        if not tab:
            raise KeyError(f"No entry found for id {item}")

        result = []

        for row in tab:
            model_name, weights_file, epsion, attack_type, extent, data_idx, data_file, data_blob, rr_blob = row

            data_noise = self.load_tensor(data_blob, DATA_SHAPE) if data_blob else None
            rr_noise = self.load_tensor(rr_blob, RR_SHAPE) if rr_blob else None
            entry = AttackEntry(
                model_name=model_name,
                weights_file=weights_file,
                epsilon=epsion,
                attack_type=attack_type,
                extent=extent,
                data_idx=data_idx,
                data_file=data_file,
                data_noise=data_noise,
                rr_noise=rr_noise
            )
            result.append(entry)


        if len(result) == 1:
            return result[0]
        return result


    def load_tensor(self, blob: bytes, shape, dtype="float32"):
        arr = np.frombuffer(blob, dtype=dtype).copy()
        return torch.from_numpy(arr).reshape(shape)

    def close(self):
        self.conn.close()


# Create a global instance of the AttackDatabase
DATABASE = AttackDatabase(os.path.join(os.path.dirname(__file__), 'Database', 'attacks.sqlite'))
