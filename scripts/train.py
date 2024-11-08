from olv_object_detection.types.yolov5.train import train
from pathlib import Path

dataset_path = Path("/home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/data/cropped_data")
models_path = Path("/home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/data/models")

annotations_path = dataset_path / "annotations.json"
images_path = dataset_path / "images"

train(coco_path=dataset_path, save_dir=models_path, epochs=200)