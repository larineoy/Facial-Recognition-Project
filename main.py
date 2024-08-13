from utils import load_image
import database as db
import project_class as pc
from pathlib import Path
from database import plot_image_with_detections

database = db.Database()

group_pic = list(Path("./pics/group/").glob("*.jpg"))[0]
duncan_pics = list(Path("./pics/TimDuncan/").glob("*.jpg"))
parker_pics = list(Path("./pics/TonyParker/").glob("*.jpg"))
