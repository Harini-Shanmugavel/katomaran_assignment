"""SQLite storage for faces (identity + embedding) and entry/exit events.

WAL mode + autocommit: every write is durable immediately, so an unexpected
interruption loses at most the statement in flight.
"""
import os
import sqlite3
import threading
from typing import List, Optional, Tuple

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    face_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    first_seen  TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    image_path  TEXT
);
CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id     INTEGER NOT NULL REFERENCES faces(face_id),
    event_type  TEXT NOT NULL CHECK (event_type IN ('entry','exit')),
    timestamp   TEXT NOT NULL,
    image_path  TEXT,
    track_id    INTEGER,
    frame_index INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_face ON events(face_id);
"""


class Database:
    def __init__(self, path: str, reset: bool = False):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if reset:
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    # ---- writes -------------------------------------------------------
    def add_face(self, timestamp: str, embedding: np.ndarray, image_path: Optional[str] = None) -> int:
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO faces (first_seen, embedding, image_path) VALUES (?,?,?)",
                (timestamp, blob, image_path),
            )
            return cur.lastrowid

    def set_face_image(self, face_id: int, image_path: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE faces SET image_path=? WHERE face_id=?", (image_path, face_id))

    def add_event(self, face_id: int, event_type: str, timestamp: str, image_path: Optional[str],
                  track_id: Optional[int], frame_index: Optional[int]) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO events (face_id, event_type, timestamp, image_path, track_id, frame_index) "
                "VALUES (?,?,?,?,?,?)",
                (face_id, event_type, timestamp, image_path, track_id, frame_index),
            )
            return cur.lastrowid

    # ---- reads --------------------------------------------------------
    def load_faces(self) -> List[Tuple[int, np.ndarray]]:
        with self._lock:
            rows = self.conn.execute("SELECT face_id, embedding FROM faces ORDER BY face_id").fetchall()
        return [(fid, np.frombuffer(blob, dtype=np.float32).copy()) for fid, blob in rows]

    def unique_count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def event_counts(self) -> dict:
        with self._lock:
            rows = self.conn.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type").fetchall()
        return dict(rows)

    def recent_events(self, limit: int = 20):
        with self._lock:
            return self.conn.execute(
                "SELECT event_id, face_id, event_type, timestamp, image_path FROM events "
                "ORDER BY event_id DESC LIMIT ?", (limit,)
            ).fetchall()

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                self.conn.close()
