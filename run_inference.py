import argparse
import json
import os
import random
import time
from typing import List, Dict


ANIMAL_CLASSES = ['dog', 'cat', 'horse', 'sheep', 'cow']
POTHOLE_CLASSES = ['pothole']  # Adjust based on your model's class names
GARBAGE_CLASSES = ['garbage', 'litter', 'trash']  # Adjust based on your model's class names


def load_model(model_name: str):
    if model_name not in ['animals', 'potholes', 'garbage']:
        raise ValueError(f"Model '{model_name}' not supported. Supported models: animals, potholes, garbage")

    # Lazy import so non-YOLO paths wouldn't require ultralytics
    from ultralytics import YOLO

    # Determine model path and classes based on model name
    if model_name == 'animals':
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'animals', 'yolov8n.pt'))
        target_classes = ANIMAL_CLASSES
    elif model_name == 'potholes':
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'potholes', 'best.pt'))
        target_classes = POTHOLE_CLASSES
    elif model_name == 'garbage':
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'garbage', 'best_model_garbge with water logging.pth'))
        target_classes = GARBAGE_CLASSES

    # Load model with error handling
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        # Return a function that returns empty results
        def _error_infer(image_paths: List[str]) -> List[Dict]:
            return [{'frame_path': img, 'detections': [], 'inference_time_ms': 0, 'error': str(e)} for img in image_paths]
        return _error_infer

    def _model_infer(image_paths: List[str]) -> List[Dict]:
        results_out: List[Dict] = []
        for img in image_paths:
            start = time.time()
            try:
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
                        if class_name in target_classes:
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
            except Exception as e:
                print(f"Error during inference for {model_name}: {e}")
                detections = []
            
            elapsed_ms = int((time.time() - start) * 1000)
            results_out.append({
                'frame_path': img,
                'detections': detections,
                'inference_time_ms': elapsed_ms,
            })
        return results_out

    return _model_infer


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


