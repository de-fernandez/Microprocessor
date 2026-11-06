# Unified Solder Joint Detection System

This version combines the roles of the previous `app.py` and `server.py` into one `app.py`.

## Files

- `app.py` — loads the YOLOv11 `.pt` model, performs inference, and provides the FastAPI endpoints.
- `index.html` — browser dashboard for the webcam and ESP32 feed.
- `requirements.txt` — Python dependencies.
- `weights.pt` — your trained model (not included; add it yourself).

## 1. Install

Use Python 3.10–3.12 and create a virtual environment.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Add your model

Copy your trained Roboflow YOLOv11 model into this folder and name it `weights.pt`.

Alternatively, set `MODEL_PATH` to the model's path.

The trained weights are not included in this package because they were not provided.

## 3. Start the system

```bash
python app.py
```

Or:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

For another device on the same Wi-Fi, use the computer's LAN IP, for example `http://192.168.1.10:8000`. Allow port 8000 through the computer firewall if needed.

## 4. Data flow

### Laptop webcam
The browser captures a frame and sends its JPEG data to `POST /predict`. `app.py` runs YOLO inference and returns detection boxes and class counts. The dashboard draws the boxes and displays Good/Bad status.

### ESP32 camera
The ESP32 firmware must POST raw JPEG bytes to:

```text
http://<COMPUTER-IP>:8000/predict?source=esp32
```

Use `Content-Type: image/jpeg`. Optional headers supported by the backend:

- `X-Device-IP`
- `X-RSSI`
- `X-Frame`
- `X-Last-Upload-Ms`
- `X-Uptime-S`

The dashboard polls `/latest`, `/latest.jpg`, and `/events` for the ESP32 feed and event log.

## 5. API endpoints

- `GET /` — dashboard
- `GET /health` — model and class information
- `POST /predict` — image inference
- `GET /latest` — latest ESP32 result
- `GET /latest.jpg` — latest annotated ESP32 frame
- `GET /events?after=0` — ESP32 event history

## 6. Notes

- This unified version uses the `.pt` model directly through Ultralytics. It does not require ONNX export or a separate `server.py`.
- The ESP32 firmware is not included because its original source was not provided. The firmware must implement the JPEG POST described above.
- This package does not include `weights.pt`; copy your trained model into the project folder before running.
- The class names are read from the model. The dashboard/backend treats class names containing `good` (case-insensitive) as good joints; other detected classes are counted as defects. Make sure your model's class names follow this convention or adjust the status logic in `app.py` and `index.html`.
