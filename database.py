import numpy as np
import pickle
import utils
from typing import Dict
from project_class import Profile
from typing import Optional
from utils import get_detection_boxes, facenet
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from facenet_models import FacenetModel

class Database:
    def __init__(self):
        # self.data = None
        # self.data = np.ndarray()
        self.data: Dict[str, Profile] = {}
        pass

    def __repr__(self):
        # makes printing the database more informative
        return f"Database({repr(self.data)})"

    def save_db(self, path):
        with open(path, mode="wb") as file:
            pickle.dump(self.data, file)

    def load_db(self, path):
        with open(path, "rb") as file:
            self.data = pickle.load(file)

