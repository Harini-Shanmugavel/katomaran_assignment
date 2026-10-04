"""Single place that writes an event to all three sinks: image folder, database, events.log.

Order matters for resilience: image first, then DB row, then the log line.
"""
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

import cv2
import numpy as np

from .database import Database


class EventLogger:
    def __init__(self, cfg, db: Database):
        self.db = db
        self.logs_dir = cfg.storage.logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        log_file = cfg.storage.log_file
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

        self.log = logging.getLogger("face_tracker")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        if not self.log.handlers:  # avoid duplicate handlers on re-init
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            fh = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.log.addHandler(fh)
            self.log.addHandler(sh)

    # ---- helpers ------------------------------------------------------
    def info(self, msg: str, *args) -> None:
        self.log.info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        self.log.warning(msg, *args)

    def _save_crop(self, subdir: str, face_id: int, ts: datetime, crop: Optional[np.ndarray]) -> Optional[str]:
        if crop is None or crop.size == 0:
            return None
        folder = os.path.join(self.logs_dir, subdir, ts.strftime("%Y-%m-%d"))
        os.makedirs(folder, exist_ok=True)
        name = f"face_{face_id:04d}_{ts.strftime('%H%M%S_%f')[:-3]}.jpg"
        path = os.path.join(folder, name)
        cv2.imwrite(path, crop)
        return path.replace("\\", "/")

    # ---- public API ---------------------------------------------------
    def register_face(self, ts: datetime, embedding: np.ndarray, crop: Optional[np.ndarray], track_id: int) -> int:
        face_id = self.db.add_face(ts.isoformat(timespec="milliseconds"), embedding, None)
        path = self._save_crop("registered", face_id, ts, crop)
        if path:
            self.db.set_face_image(face_id, path)
        self.log.info("REGISTERED face_id=%d track=%d embedding_dim=%d image=%s",
                      face_id, track_id, len(embedding), path)
        return face_id

    def log_event(self, event_type: str, face_id: int, ts: datetime, crop: Optional[np.ndarray],
                  track_id: int, frame_index: int) -> None:
        subdir = "entries" if event_type == "entry" else "exits"
        path = self._save_crop(subdir, face_id, ts, crop)
        self.db.add_event(face_id, event_type, ts.isoformat(timespec="milliseconds"), path, track_id, frame_index)
        self.log.info("%s face_id=%d track=%d frame=%d event_ts=%s image=%s",
                      event_type.upper(), face_id, track_id, frame_index,
                      ts.isoformat(timespec="milliseconds"), path)
