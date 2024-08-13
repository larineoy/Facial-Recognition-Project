from utils import load_image
import database as db
import project_class as pc
from pathlib import Path
from database import plot_image_with_detections

database = db.Database()

group_pic = list(Path("./pics/group/").glob("*.jpg"))[0]
duncan_pics = list(Path("./pics/TimDuncan/").glob("*.jpg"))
parker_pics = list(Path("./pics/TonyParker/").glob("*.jpg"))

for picture_path in duncan_pics:
    image = load_image(picture_path)
    database.add_image_to_database("Tim Duncan", image)

for pic_path in parker_pics:
    img = load_image(pic_path)
    database.add_image_to_database("Tony Parker", img)

img = load_image(group_pic)
plot_image_with_detections(img, database)
