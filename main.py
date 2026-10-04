"""Entry point.

Examples (run from the project folder):
    python main.py                                  # uses config.json (video file)
    python main.py --source data/sample.mp4 --show  # live preview window
    python main.py --source rtsp://user:pass@ip:554/stream
    python main.py --reset-db                       # start with an empty database
"""
import argparse
import signal

import cv2

from face_tracker.config import load_config
from face_tracker.database import Database
from face_tracker.event_logger import EventLogger
from face_tracker.models import ArcFaceRecognizer, YoloFaceDetector
from face_tracker.pipeline import FaceTrackingPipeline
from face_tracker.video import draw_overlay, frame_source


def parse_args():
    p = argparse.ArgumentParser(description="Intelligent face tracker + unique visitor counter")
    p.add_argument("--config", default="config.json")
    p.add_argument("--source", help="video file path or rtsp:// URL (overrides config)")
    p.add_argument("--show", action="store_true", help="show a live preview window (press q to quit)")
    p.add_argument("--save-video", help="write an annotated video to this path")
    p.add_argument("--reset-db", action="store_true", help="delete the existing database first")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg.source = args.source
    if args.show:
        cfg.output.show_window = True
    if args.save_video:
        cfg.output.save_annotated_video = args.save_video
    if args.reset_db:
        cfg.storage.reset_db_on_start = True

    db = Database(cfg.storage.db_path, reset=cfg.storage.reset_db_on_start)
    ev = EventLogger(cfg, db)
    ev.info("CONFIG source=%s skip_frames=%d", cfg.source, cfg.detection.skip_frames)

    stopping = {"flag": False}

    def request_stop(*_):
        stopping["flag"] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    pipeline = FaceTrackingPipeline(cfg, YoloFaceDetector(cfg), ArcFaceRecognizer(cfg), db, ev)
    writer = None
    try:
        for frame, ts in frame_source(cfg, ev, lambda: stopping["flag"]):
            tracks = pipeline.process_frame(frame, ts)
            if cfg.output.show_window or cfg.output.save_annotated_video:
                vis = draw_overlay(frame.copy(), tracks, db.unique_count())
                if cfg.output.save_annotated_video and writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(cfg.output.save_annotated_video,
                                             cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                if writer is not None:
                    writer.write(vis)
                if cfg.output.show_window:
                    h,w=vis.shape[:2]; max_w,max_h=1200,700; scale=min(max_w/w,max_h/h,1.0); display=cv2.resize(vis,(int(w*scale),int(h*scale))) if scale<1 else vis; cv2.imshow("Face Tracker",display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        pipeline.finalize()  # guarantees an exit event for every open entry
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nUnique visitors: {db.unique_count()}")
        db.close()


if __name__ == "__main__":
    main()

