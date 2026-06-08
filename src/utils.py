import numpy as np
import random
import torch
import os
import json
import cv2

def preprocess_input(data_chunk):
    
    data = np.array(data_chunk, dtype=np.float32)

    if data.size == 784:
        data = data.reshape(28,28)

    data = data/255.0
    data = np.expand_dims(data, axis=(0,1))
    return torch.tensor(data, dtype=torch.float32)

def save_metrics(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    metrics_to_save = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        metrics_to_save[key] = float(value)
    with open(save_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)

def seed_everything(seed=42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_ui_image(opencv_image):

    if len(opencv_image.shape) == 3:
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = opencv_image
    
    coords = cv2.findNonZero(gray)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        cropped = gray[y:y+h, x:x+w]

        resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)

        canvas28 = np.zeros((28, 28), dtype=np.uint8)
        canvas28[4:24, 4:24] = resized
    else:
        canvas28 = np.zeros((28, 28), dtype=np.uint8)

    normalized = canvas28.astype(np.float32) / 255.0
    tensor_data = np.expand_dims(normalized, axis=(0,1))

    return torch.tensor(tensor_data, dtype=torch.float32)