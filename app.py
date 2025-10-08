import json
import threading
from flask import Flask, request, jsonify

from stream_processor import StreamProcessor


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


@app.route('/start_stream', methods=['POST'])
def start_stream():
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


