import json

FP = "/home/pupil/Documents/upgrad/msc/datasets/lisa/"
TRAIN_PATH = FP +"train.json"
VAL_PATH = FP+"val.json"

COMB_PATH = FP+"train_val_true.json"

def read_json(fp):
    with open(fp,"r") as stream:
        p = json.load(stream)
    return p

train_annos = read_json(TRAIN_PATH)
val_anno = read_json(VAL_PATH)

train_val_anno = {
    "images": train_annos["images"] + val_anno["images"],
    "annotations": train_annos["annotations"] + val_anno["annotations"],
    "categories": train_annos["categories"],

}


with open(COMB_PATH,"w") as stream:
    json.dump(train_val_anno,stream)
