"""This module is to split a small validation set from the combined train_val dataset(with all 7 classes)
"""
import json
import os
import random
import shutil

BASE_PATH = "/home/pupil/Documents/upgrad/msc/datasets/lisa/"
JP = BASE_PATH + "train_val_true.json"
IMGP = BASE_PATH + "train_val"
VAL_FP = BASE_PATH + "train_val_val"
VAL_JP = BASE_PATH + "train_val_true_val.json"

random.seed(100)
SPLIT = 0.80

def read_json(fp):
    with open(fp) as stream:
        return json.load(stream)

def write_json(annos,fp):
    with open(fp,"w") as stream:
        json.dump(annos,stream)

def split(labels):
    train_imgs = random.sample(labels["images"], int(SPLIT * len(labels["images"])))
    train_img_ids = {y["id"] for y in train_imgs}
    test_img_ids = set()

    val_imgs, val_imgnames = [],[]
    for imgd in labels["images"]:
        if imgd["id"] not in train_img_ids:
            test_img_ids.add(imgd["id"])
            val_imgs.append(imgd)
            val_imgnames.append(imgd["file_name"])

    train_annos ,test_annos = [],[]

    for anno in labels["annotations"]:
        if anno["image_id"] in test_img_ids:
            test_annos.append(anno)
        else:
            train_annos.append(anno)

    train_labels = {
        "categories": labels["categories"],
        "images": train_imgs,
        "annotations" : train_annos
    }

    test_labels = {
        "categories": labels["categories"],
        "images": val_imgs,
        "annotations" : test_annos
    }

    return train_labels, test_labels, val_imgnames

def copyfiles(sfp,dfp,imgs):
    os.makedirs(dfp,exist_ok=True)
    for imgp in imgs:
        realpath = os.path.join(sfp,imgp)
        shutil.copy(realpath,dfp)

def main(source_json):
    labels = read_json(source_json)
    tr_labels, ts_labels, ts_imgfps = split(labels)
    copyfiles(IMGP, VAL_FP,ts_imgfps)

    write_json(tr_labels,BASE_PATH+"train_val_true_train.json")
    write_json(ts_labels,VAL_JP)




if __name__ == '__main__':
    main(JP)
