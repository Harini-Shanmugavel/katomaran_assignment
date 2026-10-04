"""Download the YOLOv8 face weights into models/. (InsightFace downloads its own models on first run.)"""
import os
import urllib.request

URL = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
DEST = os.path.join("models", "yolov8n-face.pt")

os.makedirs("models", exist_ok=True)
if os.path.exists(DEST):
    print("Already downloaded:", DEST)
else:
    print("Downloading", URL)
    urllib.request.urlretrieve(URL, DEST)
    print("Saved to", DEST, f"({os.path.getsize(DEST) / 1e6:.1f} MB)")
