import csv
import json
import os
import threading
from flask import Flask, request, jsonify

from stream_processor import StreamProcessor, CAMERA_DATA_FILE


app = Flask(__name__)

# In-memory registry of running processors keyed by camera_id
running_processors_lock = threading.Lock()
running_processors = {}


@app.route('/')
def root():
    # Serve the simple form
    return app.send_static_file('index.html')


@app.route('/favicon.ico')
def favicon():
    # No favicon provided; avoid log noise
    return ('', 204)


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get list of all cameras from CSV"""
    cameras = []
    try:
        if os.path.exists(CAMERA_DATA_FILE):
            with open(CAMERA_DATA_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cameras.append({
                        'cameraID': row.get('cameraID', ''),
                        'latitude': row.get('latitude', ''),
                        'longitude': row.get('longitude', '')
                    })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"cameras": cameras}), 200


@app.route('/api/cameras', methods=['POST'])
def add_camera():
    """Add a new camera to CSV"""
    data = request.get_json(silent=True) or {}
    camera_id = data.get('cameraID')
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    # Strip whitespace and validate
    camera_id = str(camera_id).strip() if camera_id else ''
    
    if not camera_id or latitude is None or longitude is None:
        return jsonify({"error": "Missing cameraID, latitude, or longitude"}), 400

    try:
        # Check if camera already exists
        cameras = []
        file_exists = os.path.exists(CAMERA_DATA_FILE)
        if file_exists:
            with open(CAMERA_DATA_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                cameras = list(reader)
                for row in cameras:
                    if str(row.get('cameraID')).strip() == camera_id:
                        return jsonify({"error": "Camera ID already exists"}), 400

        # Append new camera
        with open(CAMERA_DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['cameraID', 'latitude', 'longitude'])
            writer.writerow([camera_id, latitude, longitude])

        return jsonify({"status": "added", "cameraID": camera_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/start_streams', methods=['POST'])
def start_streams():
    """Start multiple RTSP streams simultaneously"""
    data = request.get_json(silent=True) or {}
    streams = data.get('streams', [])
    model_name = data.get('model_name', 'animals')

    if not streams or not isinstance(streams, list):
        return jsonify({"error": "Missing or invalid 'streams' array"}), 400

    results = []
    with running_processors_lock:
        for stream in streams:
            rtsp_url = stream.get('rtsp_url')
            camera_id = stream.get('camera_id')

            if not rtsp_url or not camera_id:
                results.append({
                    "camera_id": camera_id,
                    "status": "error",
                    "message": "Missing rtsp_url or camera_id"
                })
                continue

            camera_id_str = str(camera_id)
            if camera_id_str in running_processors and running_processors[camera_id_str].is_alive():
                results.append({
                    "camera_id": camera_id_str,
                    "status": "already_running"
                })
                continue

            try:
                processor = StreamProcessor(rtsp_url=rtsp_url, camera_id=camera_id_str, model_name=str(model_name))
                processor.daemon = True
                processor.start()
                running_processors[camera_id_str] = processor
                results.append({
                    "camera_id": camera_id_str,
                    "status": "started"
                })
            except Exception as e:
                results.append({
                    "camera_id": camera_id_str,
                    "status": "error",
                    "message": str(e)
                })

    return jsonify({"results": results, "model_name": model_name}), 200


@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Legacy endpoint for single stream (backward compatibility)"""
    data = request.get_json(silent=True) or {}
    rtsp_url = data.get('rtsp_url')
    camera_id = data.get('camera_id')
    model_name = data.get('model_name', 'animals')

    if not rtsp_url or not camera_id:
        return jsonify({"error": "Missing 'rtsp_url' or 'camera_id'"}), 400

    with running_processors_lock:
        if camera_id in running_processors and running_processors[camera_id].is_alive():
            return jsonify({"status": "already_running", "camera_id": camera_id}), 200

        processor = StreamProcessor(rtsp_url=rtsp_url, camera_id=str(camera_id), model_name=str(model_name))
        processor.daemon = True
        processor.start()
        running_processors[camera_id] = processor

    return jsonify({"status": "started", "camera_id": camera_id, "model_name": model_name}), 200


if __name__ == '__main__':
    # Basic dev server; production should use a proper WSGI server
    app.run(host='0.0.0.0', port=5000, debug=True)


