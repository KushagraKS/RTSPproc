#!/usr/bin/env python3
import os
import random
import numpy as np
import torch
from ultralytics import YOLO

# --- Ensure reproducibility ---
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic convolutions
    torch.backends.cudnn.benchmark = False     # Disable auto-tuner to ensure reproducibility

set_seed(0)

# --- Training Configuration ---
run_name = 'yolo1m_pretrained'
model = YOLO('/data/sree/gis/New pothole detection.v2i.yolov12/yolo12m.pt')

training_args = {
    'data': '/data/sree/gis/pothole deteion new/data.yaml',
    'epochs': 400,
    'batch': 24,
    'imgsz': 640,
    'device': 0,
    'workers': 8,
    'patience': 20,
    'save': True,
    'save_period': 10,
    'cache': False,
    'project': 'runs_yolo12m/detect',
    'name': f'version_12m_pretrained',
    'exist_ok': True,
    'pretrained': False,
    'optimizer': 'auto',
    'verbose': True,
    'seed': 0,
    'single_cls': False,
    'cos_lr': True,
    'resume': False,
    'amp': True,
    'val': True,
    'split': 'val',
    'plots': True
}

model.train(**training_args)
metrics = model.val()
print(f"v12 • {run_name} metrics:", metrics)