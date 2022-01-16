OD_IMAGE_FILES = [
    "dog.jpg",
    "eagle.jpg",
    "giraffe.jpg",
    "horses.jpg",
    "kite.jpg",
    "person.jpg"
]

IMAGE_PATH = "./images"
OUTPUT_PATH = "./output"

from pathlib import Path
OD_IMAGE_PATHS = [Path(IMAGE_PATH) / image_file for image_file in OD_IMAGE_FILES]
OD_OUTPUT_PATHS = [{"scores": Path(OUTPUT_PATH) / f"{image_file}.scores.npy", "boxes": Path(OUTPUT_PATH) / f"{image_file}.boxes.npy"} for image_file in OD_IMAGE_FILES]