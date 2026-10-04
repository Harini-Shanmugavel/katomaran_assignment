"""Video input (file or RTSP, with auto-reconnect) and the on-screen overlay."""
import os
import time
from datetime import datetime, timedelta
from typing import Iterator, Tuple

import cv2
import numpy as np


def is_stream(source: str) -> bool:
    return str(source).lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


def frame_source(cfg, ev, stop_flag) -> Iterator[Tuple[np.ndarray, datetime]]:
    """Yield (frame, timestamp). Files use start_time + position (video time);
    streams use the wall clock. RTSP streams reconnect automatically."""
    source = cfg.source
    stream = is_stream(source)
    if stream and cfg.stream.rtsp_tcp:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    start = datetime.now()
    attempts = 0
    frame_no = 0
    while not stop_flag():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            attempts += 1
            ev.warning("SOURCE_OPEN_FAILED source=%s attempt=%d", source, attempts)
            if not stream or (cfg.stream.max_reconnect_attempts and attempts >= cfg.stream.max_reconnect_attempts):
                return
            time.sleep(cfg.stream.reconnect_delay_seconds)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        ev.info("SOURCE_OPENED source=%s fps=%.1f stream=%s", source, fps, stream)
        attempts = 0
        while not stop_flag():
            ok, frame = cap.read()
            if not ok:
                break
            ts = datetime.now() if stream else start + timedelta(seconds=frame_no / fps)
            frame_no += 1
            yield frame, ts
        cap.release()
        if not stream:
            return  # end of file
        ev.warning("STREAM_INTERRUPTED, reconnecting in %ss", cfg.stream.reconnect_delay_seconds)
        time.sleep(cfg.stream.reconnect_delay_seconds)


def draw_overlay(frame: np.ndarray, tracks, unique_count: int) -> np.ndarray:
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.box]
        color = (0, 200, 0) if t.face_id is not None else (0, 200, 255)
        label = f"ID {t.face_id}" if t.face_id is not None else f"track {t.track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Unique visitors: {unique_count}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

