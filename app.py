"""
Unified Solder Joint Detection System
- One app.py handles YOLOv11 inference and the FastAPI web backend.
- index.html is served by this application.
- Accepts browser webcam frames and ESP32 JPEG uploads.

Setup:
    pip install -r requirements.txt
    # Put your trained model at ./weights.pt
    uvicorn app:app --host 0.0.0.0 --port 8000

Open http://localhost:8000
"""

import os
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "weights.pt")))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU = 0.50

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"YOLO weights not found: {MODEL_PATH}\n"
        "Place weights.pt beside app.py or set the MODEL_PATH environment variable."
    )

# Load the model once when the server starts, not once per request.
model = YOLO(str(MODEL_PATH))
MODEL_NAMES = model.names
inference_lock = threading.Lock()

app = FastAPI(title="Solder Joint Detection System", version="1.0.0")

latest_lock = threading.Lock()
latest_frame = None
latest_result = None
latest_id = 0
latest_timestamp = None
event_log = deque(maxlen=20)


def infer_image(image_bytes: bytes, confidence: float, iou: float) -> tuple[dict, bytes]:
    """Decode an image, run YOLO, draw detections, and return JSON-ready results."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image. Send JPEG or PNG bytes.")

    height, width = frame.shape[:2]
    started = time.perf_counter()

    # Ultralytics handles preprocessing, model inference, and NMS consistently.
    with inference_lock:
        prediction = model.predict(
            source=frame,
            conf=confidence,
            iou=iou,
            verbose=False
        )[0]

    inference_ms = (time.perf_counter() - started) * 1000
    counts = {str(name): 0 for name in MODEL_NAMES.values()}
    boxes_json = []
    annotated = frame.copy()

    if prediction.boxes is not None:
        xyxy = prediction.boxes.xyxy.cpu().numpy()
        class_ids = prediction.boxes.cls.cpu().numpy().astype(int)
        scores = prediction.boxes.conf.cpu().numpy()

        for coords, class_id, score in zip(xyxy, class_ids, scores):
            x0, y0, x1, y1 = [int(value) for value in coords]
            class_name = str(MODEL_NAMES.get(int(class_id), f"class_{class_id}"))
            counts[class_name] = counts.get(class_name, 0) + 1

            is_good = "good" in class_name.lower()
            color = (50, 190, 80) if is_good else (40, 40, 230)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
            label = f"{class_name} {float(score):.0%}"
            label_y = max(22, y0 - 8)
            cv2.putText(
                annotated, label, (x0, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )

            boxes_json.append({
                "class": class_name,
                "conf": round(float(score), 4),
                "x0": x0, "y0": y0, "x1": x1, "y1": y1
            })

    bad_count = sum(
        count for name, count in counts.items()
        if "good" not in name.lower()
    )
    good_count = sum(
        count for name, count in counts.items()
        if "good" in name.lower()
    )
    status = "BAD" if bad_count else ("GOOD" if good_count else "NO_JOINTS")

    result = {
        "status": status,
        "good": good_count,
        "bad": bad_count,
        "counts": counts,
        "boxes": boxes_json,
        "width": width,
        "height": height,
        "inference_ms": round(inference_ms, 2)
    }

    ok, encoded = cv2.imencode(".jpg", annotated)
    annotated_jpeg = encoded.tobytes() if ok else image_bytes
    return result, annotated_jpeg


@app.get("/")
def home():
    page = BASE_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="index.html is missing.")
    return FileResponse(page)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_PATH.name,
        "classes": list(MODEL_NAMES.values())
    }


@app.post("/predict")
async def predict(
    request: Request,
    conf: float = Query(DEFAULT_CONFIDENCE, ge=0.01, le=0.99),
    iou: float = Query(DEFAULT_IOU, ge=0.05, le=0.95),
    source: str = Query("browser")
):
    global latest_frame, latest_result, latest_id, latest_timestamp

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Image body is empty.")
    if len(body) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    result, annotated_jpeg = await run_in_threadpool(infer_image, body, conf, iou)

    if source.lower() == "esp32":
        device = {
            "ip": request.headers.get("x-device-ip"),
            "rssi": request.headers.get("x-rssi"),
            "frame": request.headers.get("x-frame"),
            "upload_ms": request.headers.get("x-last-upload-ms"),
            "uptime_s": request.headers.get("x-uptime-s")
        }
        with latest_lock:
            latest_id += 1
            latest_timestamp = time.time()
            latest_frame = annotated_jpeg
            latest_result = dict(result)
            latest_result["id"] = latest_id
            latest_result["device"] = device
            event_log.append({
                "id": latest_id,
                "time": latest_timestamp,
                "status": result["status"],
                "good": result["good"],
                "bad": result["bad"],
                "counts": result["counts"],
                "inference_ms": result["inference_ms"],
                "device": device
            })
        result["id"] = latest_id
        result["device"] = device

    return result


@app.get("/latest")
def get_latest():
    with latest_lock:
        if latest_result is None:
            return {"id": 0, "result": None, "age_s": None}
        return {
            "id": latest_id,
            "result": latest_result,
            "age_s": round(time.time() - latest_timestamp, 2)
        }


@app.get("/latest.jpg")
def get_latest_jpg():
    with latest_lock:
        frame = latest_frame
    if frame is None:
        return JSONResponse({"error": "No ESP32 frame received yet."}, status_code=404)
    return Response(frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/events")
def get_events(after: int = 0):
    with latest_lock:
        return {"events": [event for event in event_log if event["id"] > after]}


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    url = "http://localhost:8000/"

    # Start the browser after the server begins running.
    def open_browser():
        import time
        time.sleep(2)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    # Start the FastAPI server.
    uvicorn.run(app, host="0.0.0.0", port=8000)
