import cv2
import torch
import json

PFP = "/home/pupil/Documents/upgrad/msc/apm_output/inference/proposal_predictions.pth"
JFP = "/home/pupil/Documents/upgrad/msc/datasets/lisa/train_val.json"
hp = "/home/pupil/Documents/upgrad/msc/datasets/lisa/train_val/"
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

imgs = props["ids"][:N]
propos = props["boxes"][:N]

imgp = []

n = N
for imgd in annos["images"]:
    if imgd["id"] in imgs:
        imgp.append(imgd["file_name"])
        n -= 1
    if n == 0:
        break


for fp,propo in zip(imgp,propos):
    draw_prop(fp,propo,f"{fp}")
