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

    def add_image_to_database(self, name, image_array: np.ndarray):
        """
        Given the name of a person and a picture containing only that
        person's face, adds a descriptor vector to that person's profile

        Parameters
        ----------
        name : str

        array : np.ndarray, shape-(H, W, C)
        """
        boxes = get_detection_boxes(image_array)

        print("Boxes:", boxes)

        if len(boxes) != 1:
            print(f"Issue: the image contains a picture with {len(boxes)} faces")
            print("Each image should contain exactly 1 face. Skipping this image!")
            return

        descriptor = facenet.compute_descriptors(image_array, boxes)

        profile = self.data.setdefault(name, Profile(name))
        profile.add_desc(descriptor)

        print("Image Added to Database")

    def remove_prof(self, name):
        # self.data.delete(prof)
        self.data.pop(name)

    def match_query(self, unknown_face_descriptors):
        distances = {}
        for name, profile in self.data.items():
            desc_db = profile.mean_desc()

            distances[name] = utils.distance(unknown_face_descriptors, desc_db)
        value = min(distances.values())
        if value > 0.65:
            return "Unknown"
        else:
            for key in distances.keys():
                if distances[key] == value:
                    return key


def plot_image_with_detections(
    img_array: np.ndarray, face_rec_db: Optional[Database] = None
):
    """
    Plot an image with bounding boxes included around detected faces.

    If a face-recognition database is provided, each detected face will
    be queried for a match, and the name will be included with the bounding
    box.
    """

    detection_boxes = get_detection_boxes(img_array)
    plt.imshow(img_array)
    ax = plt.gca()

    facenet = FacenetModel()
    descriptors = facenet.compute_descriptors(img_array, detection_boxes)

    for box, descriptor in zip(detection_boxes, descriptors):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        if face_rec_db is not None:
            name = face_rec_db.match_query(descriptor)
            plt.text(x1, y1 + 40, str(name), color="red")

