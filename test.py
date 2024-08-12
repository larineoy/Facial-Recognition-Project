from utils import load_image, get_detection_boxes
import database as db
import project_class as pc
from pathlib import Path

database = db.Database()

group_pic = list(Path("./pics/group/").glob("*.jpg"))[0]
duncan_pics = list(Path("./pics/TimDuncan/").glob("*.jpg"))
parker_pics = list(Path("./pics/TonyParker/").glob("*.jpg"))

img = load_image(group_pic)

for picture_path in duncan_pics:
    image = load_image(picture_path)
    database.add_image_to_database("Tim Duncan", image)


print(database)
