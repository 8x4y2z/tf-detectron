import cv2
import torch
import json

from detectron2.structures import boxes

PFP = "/home/pupil/Documents/upgrad/msc/centernet_out_new/inference/instances_predictions.pth"
JFP = "/home/pupil/Documents/upgrad/msc/datasets/lisa/val.json"
hp = "/home/pupil/Documents/upgrad/msc/datasets/lisa/val/"
N = 5


def draw_prop(img_fp,props,out_fp):
    ar = cv2.imread((hp+img_fp))
    for box in props:
        _ = cv2.rectangle(ar, tuple(int(y) for y in box[:2]), tuple(int(y) for y in box[-2:]),
                      (255,0,0),2)
        cv2.imwrite(f"{out_fp}.jpg",ar)

with open(JFP,"r") as stream:
    annos = json.load(stream)

with open(PFP,"rb") as stream:
    props = torch.load(stream)

sample_res = props[:N]
imgs = {d["image_id"] for d in sample_res}
imgid_idx_m = {d["image_id"]:idx for idx,d in enumerate(sample_res)}

imgp = []
imgfp_ids = []
n = N

for imgd in annos["images"]:
    if imgd["id"] in imgs:
        imgp.append(imgd["file_name"])
        imgfp_ids.append(imgd["id"])
        n -= 1
    if n == 0:
        break


for i,fp in enumerate(imgp):
    insts = sample_res[imgid_idx_m[imgfp_ids[i]]]["instances"]
    boxes = [d["bbox"] for d in insts]
    draw_prop(fp,boxes,f"{fp}_cen")
