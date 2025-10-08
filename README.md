# RTSP Stream Orchestrator (Flask)

A Flask orchestrator that captures frames from RTSP streams, batches them, and delegates inference to isolated, model-specific Python virtual environments. Results are appended to a JSON file with metadata and a base64 frame snapshot.

## Structure
```
/project
|-- app.py                      # Flask app: /, /start_stream
|-- stream_processor.py         # Capture + inference orchestration
|-- run_inference.py            # Standalone model runner (animals only)
|-- models/
|   |-- yolo_v8/
|   |   |-- requirements_yolo_v8.txt
|   |-- animals/
|       |-- requirements_animals.txt
|       |-- yolov8n.pt
|-- venvs/                      # Model virtualenvs (created at runtime)
|-- frames/                     # Captured frames per session
|-- static/
|   |-- index.html              # Simple UI
|-- camera_data.csv             # cameraID, latitude, longitude
|-- predictions.json            # Append-only results
|-- requirements.txt            # Orchestrator deps
|-- README.md
```

## Prerequisites
- ffmpeg for OpenCV RTSP
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Python 3.10–3.12 recommended

## Setup (orchestrator)
```bash
python -m venv /home/gravity/Documents/lx/rtsp_share/venv
/home/gravity/Documents/lx/rtsp_share/venv/bin/pip install -r /home/gravity/Documents/lx/rtsp_share/requirements.txt
```

## Run
```bash
/home/gravity/Documents/lx/rtsp_share/venv/bin/python /home/gravity/Documents/lx/rtsp_share/app.py
```
- Open `http://localhost:5000/` (serves `static/index.html`)

## Start a stream
- Web form: `http://localhost:5000/`
- cURL:
```bash
curl -X POST http://localhost:5000/start_stream \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url":"rtsp://YOUR_RTSP","camera_id":"CAM-001","model_name":"animals"}'
```
`model_name` defaults to `animals`. The current build of `run_inference.py` supports only `animals`.

## How it works
- Captures a frame every 3 seconds into `frames/<cameraID_sessionId>/`.
- When 10 frames are buffered:
  1. Ensures `venvs/<model_name>` exists and installs requirements once per venv (marker file `.deps_installed`).
  2. Runs `run_inference.py` in the model venv.
  3. Enriches results with camera location, host IP, date/time, and base64 of the first frame.
  4. Appends one JSON object per 10-frame batch to `predictions.json`.
  5. Deletes processed frames and the temp results file.

## JSON schema (per batch)
```json
{
  "basic": {
    "camID": "CAM001",
    "Latitude": 26.4499,
    "Longitude": 80.3319,
    "Date": "08/10/2025",
    "timestamp": "16:45:10",
    "IP": "192.168.1.101",
    "complaint_type": { "stray_animal": 1, "pothole": 0 },
    "frame_base64": "<base64 of first frame>"
  },
  "stray_animal": {
    "detections": [ { "class": "dog", "confidence": 0.89, "box": [120,240,180,300] } ]
  },
  "pothole": null
}
```
- `stray_animal.detections` comes from the animals model (e.g., `person`, `dog`, `cat`, ... as configured in `run_inference.py`).
- `complaint_type.stray_animal` is 1 if any detection exists in the batch; else 0.

## Models and virtualenvs
- Model deps: `models/<model>/requirements_<model>.txt`.
- First batch for a model creates `venvs/<model>` and installs deps; subsequent batches reuse it (no repeated checks once `.deps_installed` exists).
- Supported path: `animals` using `models/animals/yolov8n.pt`.

## Per-frame JSON (optional)
If you want one JSON per problematic frame, adjust `stream_processor.py` to append per detection/frame instead of once per batch.

## Troubleshooting
- RTSP failures: verify URL/credentials; ensure `ffmpeg` installed.
- Slow first batch: venv creation and pip install.
- Missing wheels (torch/vision): prefer Python 3.10–3.12.
- Empty predictions: ensure `model_name` is `animals`, frames are being written, and logs show no model errors.
