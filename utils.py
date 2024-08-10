import numpy as np
import skimage.io as io  # type: ignore
from facenet_models import FacenetModel
from pathlib import Path
from typing import Union

facenet = FacenetModel()

def load_image(path):
    if isinstance(path, str):
        path = Path(path)
    array = io.imread(str(path))
    return array

def distance(x, y):
    # x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    # y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    x_norm = x / np.linalg.norm(x, keepdims=True)
    y_norm = y / np.linalg.norm(y, keepdims=True)

    output = 1 - (x_norm @ y_norm.T)

    return output

def get_detection_boxes(img_array: np.ndarray):
    detection_boxes, probs, _ = facenet.detect(img_array)
    retained_detection = probs >= 0.95
    detection_box = detection_boxes[retained_detection]
    return detection_box
