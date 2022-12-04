# -*- coding: utf-8 -*-

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import get_detection_dataset_dicts
import os
import random

random.seed(100)
N = 10
OUT = "propsviz2"
# OUT = "ycrcb"

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

if "lisa-props" not in MetadataCatalog.list():

    register_coco_instances(
        "lisa-props",
        {},
        "datasets/lisa/train_props.json",
        "datasets/lisa/train_props"
    )
if not os.path.exists(OUT):
    os.makedirs(OUT,exist_ok=True)


metadata = MetadataCatalog.get("lisa-props")
dataset = get_detection_dataset_dicts("lisa-props")
selected = random.sample(dataset,N)

for i,datad in enumerate(selected):
    img = cv2.imread(f"/home/pupil/Documents/upgrad/msc/{datad['file_name']}")
    img = cv2.cvtColor(img,36)
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(datad)
    cv2.imwrite(f"{OUT}/sample_newst{i}.jpg",out.get_image()[:, :, ::-1])
