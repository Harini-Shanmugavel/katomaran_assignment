"""Core pipeline: detect -> track -> identify -> log one entry/exit per visitor."""
from datetime import datetime
from typing import List

import cv2
import numpy as np

from .database import Database
from .event_logger import EventLogger
from .models import crop_box
from .tracker import FaceGallery, IouTracker, Track


class FaceTrackingPipeline:
    def __init__(self, cfg, detector, recognizer, db: Database, ev: EventLogger):
        self.cfg = cfg
        self.detector = detector
        self.recognizer = recognizer
        self.db = db
        self.ev = ev

        self.skip = max(0, int(cfg.detection.skip_frames))
        self.min_size = cfg.detection.min_face_size
        self.n_samples = max(1, int(cfg.recognition.embedding_samples))
        self.min_blur = cfg.recognition.min_blur_score

        self.tracker = IouTracker(
            cfg.tracking.iou_threshold,
            cfg.tracking.exit_after_frames
        )

        self.gallery = FaceGallery(cfg.recognition.similarity_threshold)

        for face_id, emb in db.load_faces():
            self.gallery.add(face_id, emb)

        # face_id -> currently active track
        self.active_faces = {}

        # Face IDs that have already received an ENTRY event
        self.entered_faces = set()

        # Face IDs that have already received an EXIT event
        self.exited_faces = set()

        self.frame_idx = 0

        self.ev.info(
            "PIPELINE_START known_faces=%d skip_frames=%d",
            len(self.gallery.ids),
            self.skip
        )

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, ts: datetime) -> List[Track]:
        self.tracker.predict()

        if self.frame_idx % (self.skip + 1) == 0:
            dets = self.detector.detect(frame)

            dets = np.array(
                [
                    d for d in dets
                    if min(d[2] - d[0], d[3] - d[1]) >= self.min_size
                ],
                dtype=np.float32
            ).reshape(-1, 5)

            for tr in self.tracker.update(
                dets,
                ts,
                self.frame_idx
            ):
                if tr.hits == 1:
                    self.ev.info(
                        "TRACK_STARTED track=%d frame=%d",
                        tr.track_id,
                        self.frame_idx
                    )

                tr.last_crop = crop_box(frame, tr.box)

                if tr.face_id is None:
                    self._collect_sample(tr, frame)

        for lost in self.tracker.pop_lost():
            self._handle_exit(lost)

        self.frame_idx += 1

        return list(self.tracker.tracks.values())

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Close every remaining track when the stream ends."""

        for tr in list(self.tracker.tracks.values()):
            self.tracker.remove(tr)
            self._handle_exit(tr)

        self.ev.info(
            "PIPELINE_END unique_visitors=%d events=%s",
            self.db.unique_count(),
            self.db.event_counts()
        )

    # ------------------------------------------------------------------
    def _collect_sample(
        self,
        tr: Track,
        frame: np.ndarray
    ) -> None:

        crop = tr.last_crop

        if crop is None or crop.size == 0:
            return

        blur = cv2.Laplacian(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
            cv2.CV_64F
        ).var()

        if blur < self.min_blur:
            return

        emb = self.recognizer.embed(frame, tr.box)

        if emb is None:
            return

        tr.samples.append(emb)

        self.ev.info(
            "EMBEDDING track=%d sample=%d/%d",
            tr.track_id,
            len(tr.samples),
            self.n_samples
        )

        if len(tr.samples) >= self.n_samples:
            self._identify(tr)

    # ------------------------------------------------------------------
    def _identify(self, tr: Track) -> None:
        mean = np.mean(tr.samples, axis=0)

        face_id, score = self.gallery.match(mean)

        if face_id is None:
            face_id = self.ev.register_face(
                tr.created_ts,
                mean,
                tr.last_crop,
                tr.track_id
            )

            self.gallery.add(face_id, mean)

            self.ev.info(
                "IDENTITY_CREATED face_id=%d track=%d",
                face_id,
                tr.track_id
            )

        else:
            self.ev.info(
                "RECOGNIZED face_id=%d track=%d similarity=%.3f",
                face_id,
                tr.track_id,
                score
            )

        self._assign(tr, face_id)

    # ------------------------------------------------------------------
    def _assign(self, tr: Track, face_id: int) -> None:
        """
        Connect a temporary tracking ID to a permanent face ID.

        A face receives only ONE entry event during the video.
        Temporary tracker ID changes do not create another entry.
        """

        other = self.active_faces.get(face_id)

        tr.face_id = face_id

        # --------------------------------------------------------------
        # Case 1: This face is already being tracked
        # --------------------------------------------------------------
        if other is not None and other is not tr:

            if other.time_since_update > 0:
                # Old tracker lost the face, but the new tracker
                # recognized the same permanent face_id.
                other.merged = True

                self.tracker.remove(other)

                tr.entered = True
                self.active_faces[face_id] = tr

                self.ev.info(
                    "TRACK_MERGED face_id=%d old_track=%d new_track=%d",
                    face_id,
                    other.track_id,
                    tr.track_id
                )

            else:
                # Same face detected by another active track.
                tr.shadow = True

                self.ev.info(
                    "TRACK_DUPLICATE face_id=%d track=%d (ignored)",
                    face_id,
                    tr.track_id
                )

            return

        # --------------------------------------------------------------
        # Case 2: Face has already entered previously
        # --------------------------------------------------------------
        if face_id in self.entered_faces:

            tr.entered = True
            self.active_faces[face_id] = tr

            self.ev.info(
                "RECONNECTED face_id=%d track=%d (no new entry)",
                face_id,
                tr.track_id
            )

            return

        # --------------------------------------------------------------
        # Case 3: Completely new visitor
        # --------------------------------------------------------------
        self.active_faces[face_id] = tr
        tr.entered = True

        self.entered_faces.add(face_id)

        self.ev.log_event(
            "entry",
            face_id,
            tr.created_ts,
            tr.last_crop,
            tr.track_id,
            tr.first_frame
        )

    # ------------------------------------------------------------------
    def _handle_exit(self, tr: Track) -> None:

        if tr.merged or tr.shadow:
            return

        # If this track was never identified but has samples,
        # make one final recognition attempt.
        if tr.face_id is None and tr.samples:
            self._identify(tr)

            if tr.shadow:
                return

        # Track never became a known visitor.
        if tr.face_id is None:
            self.ev.info(
                "TRACK_DROPPED track=%d (never identified)",
                tr.track_id
            )
            return

        face_id = tr.face_id

        # Remove active mapping if this is still the active track.
        if self.active_faces.get(face_id) is tr:
            del self.active_faces[face_id]

        # --------------------------------------------------------------
        # IMPORTANT:
        # Only ONE exit event is allowed for each face_id.
        # --------------------------------------------------------------
        if face_id in self.exited_faces:
            self.ev.info(
                "EXIT_ALREADY_LOGGED face_id=%d track=%d",
                face_id,
                tr.track_id
            )
            return

        self.exited_faces.add(face_id)

        self.ev.log_event(
            "exit",
            face_id,
            tr.last_seen_ts,
            tr.last_crop,
            tr.track_id,
            self.frame_idx
        )