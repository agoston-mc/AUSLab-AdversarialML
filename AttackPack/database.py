import sqlite3
import torch
import json
from dataclasses import dataclass
from collections import namedtuple

@dataclass
class DataEntry:
    original: torch.Tensor

    def to_hex(self):
        return self.original.numpy().tobytes().hex()

    @staticmethod
    def from_hex(h, shape, dtype=torch.float32):
        return DataEntry(torch.frombuffer(bytes.fromhex(h), dtype=dtype).reshape(shape))

at = namedtuple("attack_tens", ["data", "rr"])

@dataclass
class RunEntry:
    epsilon: float
    attack_type: str
    attack_tens: at
    TN_change: list[DataEntry]
    TP_change: list[DataEntry]

class Database:
    def __init__(self, db_path="attack.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._setup_schema()

    def _setup_schema(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS run_entries (
                id TEXT PRIMARY KEY,
                epsilon REAL,
                attack_type TEXT,
                attack_data TEXT,
                attack_rr TEXT,
                TN_change TEXT,
                TP_change TEXT
            )
        """)
        self.conn.commit()

    def __setitem__(self, key, value):
        re = RunEntry(*value)
        self.cur.execute("""
            INSERT OR REPLACE INTO run_entries (
                id, epsilon, attack_type, attack_data, attack_rr, TN_change, TP_change
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            key,
            re.epsilon,
            re.attack_type,
            json.dumps(re.attack_tens.data.tolist()),
            json.dumps(re.attack_tens.rr.tolist()),
            json.dumps([entry.to_hex() for entry in re.TN_change]),
            json.dumps([entry.to_hex() for entry in re.TP_change]),
        ))
        self.conn.commit()

    def get(self, key, shape, dtype=torch.float32):
        self.cur.execute("SELECT * FROM run_entries WHERE id = ?", (key,))
        row = self.cur.fetchone()
        if not row:
            return None

        _, eps, atype, data_str, rr_str, tn_list, tp_list = row
        return RunEntry(
            epsilon=eps,
            attack_type=atype,
            attack_tens=at(
                data=torch.tensor(json.loads(data_str), dtype=dtype),
                rr=torch.tensor(json.loads(rr_str), dtype=dtype)
            ),
            TN_change=[DataEntry.from_hex(h, shape, dtype) for h in json.loads(tn_list)],
            TP_change=[DataEntry.from_hex(h, shape, dtype) for h in json.loads(tp_list)]
        )

    def close(self):
        self.conn.close()
