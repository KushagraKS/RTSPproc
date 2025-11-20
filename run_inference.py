import argparse
import json
import os
import random
import time
from typing import List, Dict


ANIMAL_CLASSES = ['dog', 'cat', 'horse', 'sheep', 'cow']


def load_model(model_name: str):
    if model_name != 'animals':
        raise ValueError("Only 'animals' model is supported in this build")

    # Lazy import so non-animals paths wouldn't require ultralytics (kept for clarity)
    from ultralytics import YOLO

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'animals', 'yolov8n.pt'))
    model = YOLO(model_path)

    def _animals_infer(image_paths: List[str]) -> List[Dict]:
        results_out: List[Dict] = []
        for img in image_paths:
            start = time.time()
            results = model(img)
            detections = []
            for r in results:
                names_map = r.names  # id -> name
                if getattr(r, 'boxes', None) is None:
                    continue
                for box in r.boxes:
                    try:
                        class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                    except Exception:
                        continue
                    class_name = names_map.get(class_id, str(class_id))
                    if class_name in ANIMAL_CLASSES:
                        try:
                            conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                        except Exception:
                            conf = 0.0
                        try:
                            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy)
                        except Exception:
                            xyxy = [0, 0, 0, 0]
                        detections.append({
                            'label': class_name,
                            'confidence': round(conf, 4),
                            'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        })
            elapsed_ms = int((time.time() - start) * 1000)
            results_out.append({
                'frame_path': img,
                'detections': detections,
                'inference_time_ms': elapsed_ms,
            })
        return results_out

    return _animals_infer


def main():
    parser = argparse.ArgumentParser(description='Run model inference on frames.')
    parser.add_argument('--model-name', required=True, help='Model name, e.g., yolo_v8')
    parser.add_argument('--frames', required=True, help='JSON list of frame file paths')
    parser.add_argument('--output-file', required=True, help='Path to write JSON results')
    args = parser.parse_args()

    frames = json.loads(args.frames)
    if not isinstance(frames, list):
        raise ValueError('--frames must be a JSON list of file paths')

    infer = load_model(args.model_name)
    results = infer(frames)

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()


