"""Tests the counting/entry/exit logic without any ML models.

A fake detector reports scripted boxes; a fake recognizer returns a fixed
embedding per person. Run:  python -m pytest -q   (or)   python tests/test_pipeline_logic.py
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_tracker.config import config_from_dict
from face_tracker.database import Database
from face_tracker.event_logger import EventLogger
from face_tracker.pipeline import FaceTrackingPipeline

RNG = np.random.default_rng(0)
EMB = {p: RNG.normal(size=512).astype(np.float32) for p in "AB"}


class FakeDetector:
    def __init__(self, script):  # script: frame_index -> list of (person, x)
        self.script, self.calls = script, 0
        self.frame_idx = 0

    def detect(self, frame):
        people = self.script(int(frame[0, 0, 0]))  # frame index is stored in pixel (0,0)
        return np.array([[x, 100, x + 80, 180, 0.9] for _, x in people], dtype=np.float32).reshape(-1, 5)


class FakeRecognizer:
    def __init__(self, script):
        self.script = script

    def embed(self, frame, box):
        idx = int(frame[0, 0, 0])
        for person, x in self.script(idx):
            if abs(x - box[0]) < 20:
                return EMB[person] + RNG.normal(scale=0.01, size=512).astype(np.float32)
        return None


def run(skip_frames):
    # A: frames 0-29, brief 5-frame dropout at 10-14, gone 30-99, back 100-129
    # B: frames 50-110, overlapping A's absence and return
    def script(i):
        out = []
        if (0 <= i < 30 and not 10 <= i < 15) or 100 <= i < 130:
            out.append(("A", 100))
        if 50 <= i < 110:
            out.append(("B", 400))
        return out

    tmp = tempfile.mkdtemp()
    cfg = config_from_dict({
        "detection": {"skip_frames": skip_frames, "min_face_size": 40},
        "recognition": {"embedding_samples": 3, "min_blur_score": 0, "similarity_threshold": 0.45},
        "tracking": {"exit_after_frames": 20},
        "storage": {"db_path": f"{tmp}/t.db", "logs_dir": f"{tmp}/logs", "log_file": f"{tmp}/logs/events.log"},
    })
    db = Database(cfg.storage.db_path)
    ev = EventLogger(cfg, db)
    pipe = FaceTrackingPipeline(cfg, FakeDetector(script), FakeRecognizer(script), db, ev)
    t0 = datetime(2026, 10, 1, 12, 0, 0)
    for i in range(140):
        frame = np.full((480, 640, 3), 0, dtype=np.uint8)
        frame[0, 0, 0] = i
        pipe.process_frame(frame, t0 + timedelta(seconds=i / 25))
    pipe.finalize()
    return db


def check(skip):
    db = run(skip)
    counts = db.event_counts()
    assert db.unique_count() == 2, f"unique={db.unique_count()}"       # A and B only
    assert counts.get("entry") == 3, counts                              # A, B, A again
    assert counts.get("exit") == counts.get("entry"), counts             # exactly one exit per entry
    print(f"skip_frames={skip}: unique=2 entries={counts['entry']} exits={counts['exit']}  OK")


def test_no_skip():
    check(0)


def test_skip_2():
    check(2)


if __name__ == "__main__":
    test_no_skip()
    test_skip_2()
