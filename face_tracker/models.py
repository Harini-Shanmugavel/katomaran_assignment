"""Model wrappers. Heavy libraries are imported lazily so the rest of the code
(and the unit tests) work without ultralytics / insightface installed."""
from typing import Optional

import cv2
import numpy as np


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def crop_box(frame: np.ndarray, box, margin: float = 0.0) -> np.ndarray:
    """Crop box (x1,y1,x2,y2) from frame, expanded by `margin` and clamped to the image."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return frame[y1:y2, x1:x2].copy()


class YoloFaceDetector:
    """YOLOv8 face detector. detect() returns an (N,5) array: x1,y1,x2,y2,conf."""

    def __init__(self, cfg):
        from ultralytics import YOLO
        self.device = resolve_device(cfg.device)
        self.model = YOLO(cfg.detection.model_path)
        self.conf = cfg.detection.confidence
        self.iou = cfg.detection.iou
        self.imgsz = cfg.detection.imgsz

    def detect(self, frame: np.ndarray) -> np.ndarray:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 device=self.device, half=self.device.startswith("cuda"),
                                 verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, conf]).astype(np.float32)


class ArcFaceRecognizer:
    """InsightFace (ArcFace) embeddings, 512-d, L2-normalised.

    The YOLO box is cropped with a margin; InsightFace's light detector then finds
    landmarks inside that crop so the face can be aligned before ArcFace runs.
    If no landmarks are found we fall back to a plain 112x112 resize.
    """

    def __init__(self, cfg):
        from insightface.app import FaceAnalysis
        device = resolve_device(cfg.device)
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device.startswith("cuda") else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name=cfg.recognition.model_name,
                                allowed_modules=["detection", "recognition"],
                                providers=providers)
        self.app.prepare(ctx_id=0 if device.startswith("cuda") else -1,
                         det_thresh=0.3, det_size=(160, 160))
        self.margin = cfg.recognition.crop_margin

    def embed(self, frame: np.ndarray, box) -> Optional[np.ndarray]:
        crop = crop_box(frame, box, self.margin)
        if crop.size == 0:
            return None
        faces = self.app.get(crop)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.normed_embedding
        else:
            rec = self.app.models["recognition"]
            emb = rec.get_feat(cv2.resize(crop, (112, 112))).flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        return np.asarray(emb, dtype=np.float32)
