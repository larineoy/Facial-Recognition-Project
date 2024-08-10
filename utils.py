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
