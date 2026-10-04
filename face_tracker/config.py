"""Load config.json, merge it over built-in defaults, expose attribute access."""
import copy
import json
from types import SimpleNamespace

DEFAULTS = {
    "source": "data/sample.mp4",
    "device": "auto",  # "auto" | "cpu" | "cuda"
    "detection": {
        "model_path": "models/yolov8n-face.pt",
        "confidence": 0.5,
        "iou": 0.5,
        "imgsz": 640,
        "skip_frames": 2,  # frames skipped between detection cycles
        "min_face_size": 40,  # px, smaller faces are ignored
    },
    "recognition": {
        "model_name": "buffalo_l",
        "similarity_threshold": 0.45,  # cosine similarity for "same person"
        "embedding_samples": 3,  # detections averaged before a track is identified
        "min_blur_score": 20.0,  # Laplacian variance; below = too blurry
        "crop_margin": 0.3,
    },
    "tracking": {
        "iou_threshold": 0.25,
        "exit_after_frames": 30,  # frames without a detection before "exit"
    },
    "storage": {
        "db_path": "data/face_tracker.db",
        "logs_dir": "logs",
        "log_file": "logs/events.log",
        "reset_db_on_start": False,
    },
    "stream": {
        "rtsp_tcp": True,
        "reconnect_delay_seconds": 3,
        "max_reconnect_attempts": 0,  # 0 = retry forever
    },
    "output": {"show_window": False, "save_annotated_video": None},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def load_config(path: str = "config.json") -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    return _to_ns(_merge(DEFAULTS, user_cfg))


def config_from_dict(d: dict) -> SimpleNamespace:
    """Used by tests."""
    return _to_ns(_merge(DEFAULTS, d))
