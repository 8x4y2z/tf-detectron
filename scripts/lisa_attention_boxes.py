# -*- coding: utf-8 -*-

import json
from pathlib import Path
import shutil
from functools import partial
import numpy as np
import multiprocessing

PATH = Path("../datasets/lisa").resolve()

TRAIN_ANNO = PATH/"train.json"
TRAIN_DIR = PATH/"train"

VAL_ANNO = PATH/"val.json"
VAL_DIR = PATH/"val"

ALPHA = 5

def read_json(json_file):
    with open(json_file) as stream:
        data = json.load(stream)

    return data

def copy_files(dfp,files):
    for file_ in files:
        shutil.copy(file_,dfp)



def merge_train_val():
    """
    """

    train_anno = read_json(TRAIN_ANNO)
    val_anno = read_json(VAL_ANNO)
    train_val_anno = {
        "images": train_anno["images"] + val_anno["images"],
        "annotations": train_anno["annotations"] + val_anno["annotations"],
        "categories": train_anno["categories"],

    }
    new_dir = PATH/"train_val"
    if not new_dir.exists():
        new_dir.mkdir(exist_ok=True)
    copyfunc = partial(copy_files,new_dir)
    train_val_images = list(TRAIN_DIR.iterdir()) + list(VAL_DIR.iterdir())

    pool_size = multiprocessing.cpu_count() * 2
    splits = np.array_split(train_val_images,100)

    pool = multiprocessing.Pool(processes=pool_size)
    pool.map(copyfunc,splits)
    pool.close()
    pool.join()

    print("============================ copy files done")

    modify_annos(train_val_anno["annotations"])
    print("============================ annos modified")

    return train_val_anno

def modify_annos(annos):
    for anno in annos:
        x,y,w,h = anno["bbox"]
        xc, yc = x+w/2, y+h/2
        side = max(w,h)*ALPHA
        xm,ym = xc-side/2, yc-side/2
        anno["bbox"] = [xm,ym,side,side]



def main():
    """
    """
    train_val_anno = merge_train_val()
    with open(PATH/"train_val.json","w") as stream:
        json.dump(train_val_anno,stream)

    print("Complete.............")


if __name__ == '__main__':
    main()
