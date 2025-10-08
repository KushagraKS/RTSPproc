import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
import socket
import base64

import cv2


PREDICTIONS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'predictions.json'))
CAMERA_DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'camera_data.csv'))
FRAMES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'frames'))
VENVS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'venvs'))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))


# Ensure base directories exist
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(VENVS_DIR, exist_ok=True)


# Lock to safely append to predictions.json across threads
predictions_file_lock = threading.Lock()


def read_camera_metadata(camera_id: str):
    try:
        with open(CAMERA_DATA_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get('cameraID')) == str(camera_id):
                    return {
                        'latitude': row.get('latitude'),
                        'longitude': row.get('longitude'),
                    }
    except FileNotFoundError:
        return None
    return None


def append_predictions(results):
    # results is a list of dicts (each dict matches the requested schema)
    with predictions_file_lock:
        try:
            if os.path.exists(PREDICTIONS_FILE):
                with open(PREDICTIONS_FILE, 'r') as f:
                    existing = json.load(f) or []
            else:
                existing = []
        except Exception:
            existing = []

        existing.extend(results)
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(existing, f, indent=2)


def setup_model_environment(model_name: str) -> str:
    """
    Ensure venv exists at venvs/{model_name} and required packages are installed.
    Returns path to the python executable inside that venv.
    """
    venv_path = os.path.join(VENVS_DIR, model_name)
    python_executable = os.path.join(venv_path, 'bin', 'python')
    pip_executable = os.path.join(venv_path, 'bin', 'pip')

    requirements_file = os.path.join(MODELS_DIR, model_name, f'requirements_{model_name}.txt')
    installed_marker = os.path.join(venv_path, '.deps_installed')

    # Create venv if missing
    if not os.path.exists(python_executable):
        os.makedirs(venv_path, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)

    # Install/upgrade pip and requirements only once per venv (marker file)
    if os.path.exists(requirements_file) and not os.path.exists(installed_marker):
        subprocess.run([python_executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools'], check=True)
        subprocess.run([pip_executable, 'install', '-r', requirements_file], check=True)
        try:
            with open(installed_marker, 'w') as _:
                _.write('ok')
        except Exception:
            pass

    return python_executable


class StreamProcessor(threading.Thread):
    def __init__(self, rtsp_url: str, camera_id: str, model_name: str = 'animals'):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.model_name = model_name
        self.session_id = f"{camera_id}_{uuid.uuid4().hex[:8]}"
        self.session_dir = os.path.join(FRAMES_DIR, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)

        self.frame_queue = queue.Queue()
        self._stop_event = threading.Event()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._orchestrator_thread = threading.Thread(target=self._orchestrate_inference_loop, daemon=True)
        self._model_env_ready = False

    def stop(self):
        self._stop_event.set()

    def is_alive(self):
        # alive if any internal thread running
        return super().is_alive() or self._capture_thread.is_alive() or self._orchestrator_thread.is_alive()

    def run(self):
        self._capture_thread.start()
        self._orchestrator_thread.start()
        # Keep the main thread alive until stopped
        while not self._stop_event.is_set():
            time.sleep(0.5)

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            # Sleep and retry later if stream not open
            while not self._stop_event.is_set():
                time.sleep(3)
                cap.open(self.rtsp_url)
                if cap.isOpened():
                    break

        last_capture_time = 0.0
        try:
            while not self._stop_event.is_set():
                now = time.time()
                if now - last_capture_time < 3.0:
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    time.sleep(1.0)
                    continue

                timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')
                frame_filename = f"{self.camera_id}_{timestamp}.jpg"
                frame_path = os.path.join(self.session_dir, frame_filename)
                try:
                    cv2.imwrite(frame_path, frame)
                    self.frame_queue.put(frame_path)
                    last_capture_time = now
                except Exception:
                    # If write fails, skip and continue
                    time.sleep(0.1)
        finally:
            cap.release()

    def _orchestrate_inference_loop(self):
        while not self._stop_event.is_set():
            # Wait until we have at least 10 frames
            if self.frame_queue.qsize() < 10:
                time.sleep(0.2)
                continue

            batch_paths = []
            try:
                for _ in range(10):
                    batch_paths.append(self.frame_queue.get_nowait())
            except queue.Empty:
                # If not enough frames unexpectedly, continue
                time.sleep(0.2)
                continue

            # Prepare model environment
            try:
                if not self._model_env_ready:
                    model_python = setup_model_environment(self.model_name)
                    self._model_env_ready = True
                else:
                    # Resolve python executable path without reinstall
                    venv_path = os.path.join(VENVS_DIR, self.model_name)
                    model_python = os.path.join(venv_path, 'bin', 'python')
            except subprocess.CalledProcessError:
                # On setup failure, drop batch and continue
                self._cleanup_files(batch_paths)
                continue

            # Prepare temp output file
            temp_output = os.path.join(self.session_dir, f"tmp_results_{uuid.uuid4().hex[:8]}.json")

            # Invoke subprocess
            cmd = [
                model_python,
                os.path.abspath(os.path.join(os.path.dirname(__file__), 'run_inference.py')),
                '--model-name', self.model_name,
                '--frames', json.dumps(batch_paths),
                '--output-file', temp_output,
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                # Failed inference; clean up frames and temp file if exists
                self._cleanup_files(batch_paths)
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                continue

            # Read results
            try:
                with open(temp_output, 'r') as f:
                    predictions = json.load(f)
            except Exception:
                predictions = []

            # Build requested schema per batch (one entry per batch)
            camera_meta = read_camera_metadata(self.camera_id) or {}
            has_animal = False
            animal_detections = []
            for pred in predictions:
                for det in pred.get('detections', []):
                    label = det.get('label') or det.get('class')
                    if label:
                        # Any detected label counts toward stray_animal; animals model already filters
                        has_animal = True
                        # Map to required keys
                        box = det.get('bbox') or det.get('box') or [0, 0, 0, 0]
                        animal_detections.append({
                            'class': label,
                            'confidence': float(det.get('confidence', 0.0)),
                            'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        })

            # Prepare base64 of the first frame in batch
            frame_b64 = ''
            try:
                first_frame = batch_paths[0]
                with open(first_frame, 'rb') as fb:
                    frame_b64 = base64.b64encode(fb.read()).decode('utf-8')
            except Exception:
                frame_b64 = ''

            # Time and IP metadata
            now = datetime.now()
            date_str = now.strftime('%d/%m/%Y')
            time_str = now.strftime('%H:%M:%S')
            try:
                host_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                host_ip = '127.0.0.1'

            schema_obj = {
                'basic': {
                    'camID': str(self.camera_id),
                    'Latitude': float(camera_meta.get('latitude')) if camera_meta.get('latitude') is not None else None,
                    'Longitude': float(camera_meta.get('longitude')) if camera_meta.get('longitude') is not None else None,
                    'Date': date_str,
                    'timestamp': time_str,
                    'IP': host_ip,
                    'complaint_type': {
                        'stray_animal': 1 if has_animal else 0,
                        'pothole': 0,
                    },
                    'frame_base64': frame_b64,
                },
                'stray_animal': {
                    'detections': animal_detections,
                } if animal_detections else {'detections': []},
                'pothole': None,
            }

            append_predictions([schema_obj])

            # Cleanup processed files
            self._cleanup_files(batch_paths + [temp_output])

    def _cleanup_files(self, paths):
        for p in paths:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


