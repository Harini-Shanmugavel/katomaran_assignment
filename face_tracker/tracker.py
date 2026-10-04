"""Lightweight multi-face tracker: IoU + constant-velocity prediction, Hungarian matching.

Detection runs only every (skip_frames + 1) frames, so on the frames in between the
tracker *predicts* each box forward using its last velocity. A track that gets no
detection for `max_lost` frames is considered to have left the frame.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and every box in b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    created_ts: datetime
    last_seen_ts: datetime
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    hits: int = 1
    time_since_update: int = 0
    # identity state (managed by the pipeline)
    face_id: Optional[int] = None
    entered: bool = False
    shadow: bool = False      # duplicate of a face already visible in another track
    merged: bool = False      # absorbed by a newer track of the same person
    samples: list = field(default_factory=list)
    last_crop: Optional[np.ndarray] = None
    first_frame: int = 0

    def predict(self) -> None:
        self.box = self.box + self.velocity
        self.velocity = self.velocity * 0.9
        self.time_since_update += 1

    def update(self, box: np.ndarray, ts: datetime) -> None:
        new_box = np.asarray(box[:4], dtype=np.float32)
        if self.time_since_update > 0:
            step = max(self.time_since_update, 1)
            self.velocity = 0.5 * self.velocity + 0.5 * (new_box - self.box) / step
        self.box = new_box
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_ts = ts


class IouTracker:
    def __init__(self, iou_threshold: float, max_lost: int):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def predict(self) -> None:
        for t in self.tracks.values():
            t.predict()

    def update(self, dets: np.ndarray, ts: datetime, frame_index: int) -> List[Track]:
        """Associate detections (N,5) with tracks. Returns tracks that got a detection
        this cycle (existing + newly created)."""
        updated: List[Track] = []
        ids = list(self.tracks.keys())
        matched_det = set()
        if ids and len(dets):
            pred = np.array([self.tracks[i].box for i in ids], dtype=np.float32)
            iou = iou_matrix(pred, dets[:, :4])
            rows, cols = linear_sum_assignment(1.0 - iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    tr = self.tracks[ids[r]]
                    tr.update(dets[c], ts)
                    updated.append(tr)
                    matched_det.add(c)
        for c in range(len(dets)):
            if c in matched_det:
                continue
            tr = Track(track_id=self._next_id, box=np.asarray(dets[c][:4], dtype=np.float32),
                       created_ts=ts, last_seen_ts=ts, first_frame=frame_index)
            self.tracks[tr.track_id] = tr
            self._next_id += 1
            updated.append(tr)
        return updated

    def pop_lost(self) -> List[Track]:
        lost = [t for t in self.tracks.values() if t.time_since_update > self.max_lost]
        for t in lost:
            del self.tracks[t.track_id]
        return lost

    def remove(self, track: Track) -> None:
        self.tracks.pop(track.track_id, None)


class FaceGallery:
    """In-memory store of known embeddings for fast cosine-similarity lookup."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.ids: List[int] = []
        self.matrix = np.zeros((0, 512), dtype=np.float32)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def add(self, face_id: int, embedding: np.ndarray) -> None:
        v = self._norm(embedding)[None, :]
        if self.matrix.shape[0] == 0:
            self.matrix = np.zeros((0, v.shape[1]), dtype=np.float32)
        self.ids.append(face_id)
        self.matrix = np.vstack([self.matrix, v])

    def match(self, embedding: np.ndarray):
        """Return (face_id or None, best_similarity)."""
        if not self.ids:
            return None, 0.0
        sims = self.matrix @ self._norm(embedding)
        best = int(np.argmax(sims))
        score = float(sims[best])
        return (self.ids[best] if score >= self.threshold else None), score
