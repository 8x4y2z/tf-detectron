# -*- coding: utf-8 -*-

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import get_detection_dataset_dicts
import os
import random

N = 1000
SEED = 100
random.seed(SEED)
os.makedirs("out-new", exist_ok=True)

# register_coco_instances(
#     "lisa-train",
#     {},
#     "datasets/lisa/train.json",
#     "datasets/lisa/train"
# )

# register_coco_instances(
#     "lisa-val",
#     {},
#     "datasets/lisa/val.json",
#     "datasets/lisa/val"
# )


# register_coco_instances(
#     "lisa-test",
#     {},
#     "datasets/lisa/test.json",
#     "datasets/lisa/test"
# )


register_coco_instances(
    "lisa-train-new",
    {},
    "datasets/lisa-new/val.json",
    "datasets/lisa-new/val"
)

metadata = MetadataCatalog.get("lisa-train-new")
dataset = get_detection_dataset_dicts("lisa-train-new")
for i,datad in enumerate(random.sample(dataset,N)):
    img = cv2.imread(f"/home/pupil/Documents/upgrad/msc/{datad['file_name']}")
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(datad)
    cv2.imwrite(f"out-new/{datad['file_name'].rsplit('/',1)[-1]}",out.get_image()[:, :, ::-1])
