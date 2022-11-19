# -*- coding: utf-8 -*-

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import get_detection_dataset_dicts

register_coco_instances(
    "lisa-train",
    {},
    "datasets/lisa/train.json",
    "datasets/lisa/train"
)

register_coco_instances(
    "lisa-val",
    {},
    "datasets/lisa/val.json",
    "datasets/lisa/val"
)


register_coco_instances(
    "lisa-test",
    {},
    "datasets/lisa/test.json",
    "datasets/lisa/test"
)

metadata = MetadataCatalog.get("lisa-val")
dataset = get_detection_dataset_dicts("lisa-val")
for i,datad in enumerate(dataset[:5]):
    img = cv2.imread(f"/home/pupil/projects/tf-detectron/{datad['file_name']}")
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(datad)
    cv2.imwrite(f"sample{i}.jpg",out.get_image()[:, :, ::-1])
