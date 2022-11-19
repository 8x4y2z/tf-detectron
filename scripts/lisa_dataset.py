# -*- coding: utf-8 -*-

import json
import csv
import os
from os.path import isdir, realpath
import shutil
from collections import namedtuple
from typing import Dict, List, Optional, Set
from enum import Enum
import random
import multiprocessing
from functools import partial
import numpy as np

SEED = 100
random.seed(SEED)

ANNO_PATH = "/home/pupil/projects/tf-detectron/lisa/Annotations/Annotations/"
IMG_PATH = "/home/pupil/projects/tf-detectron/lisa/"
TRAIN_VAL_SEQ = ["dayTrain","nightTrain"]
TEST_SEQ = ["daySequence1","daySequence2","nightSequence1","nightSequence2"]

TRAIN_SPLIT = 0.75




OUTPUT_PATH = "/home/pupil/projects/tf-detectron/datasets/lisa"

Entry = namedtuple("Entry",[
    'filename',
    'annotation_tag',
    'upper_left_corner_x',
    'upper_left_corner_y',
    'lower_right_corner_x',
    'lower_right_corner_y',
    'origin_file',
    'origin_frame_number',
    'origin_track',
    'origin_track_frame_number'
    ]
)

CATS_IDX_MAPPING:Dict[str,Optional[int]] = {}
CATS_META = []

TRAIN_IMAGES_IDX_MAPPING:Dict[str,Optional[int]] = {}
TRAIN_IMAGES_META = []
TRAIN_ANNO_META = []

TEST_IMAGES_IDX_MAPPING:Dict[str,Optional[int]] = {}
TEST_IMAGES_META = []
TEST_ANNO_META = []

IMAGE_COUNTER = CAT_COUNTER = ANN0_COUNTER = 0

class FLOW(Enum):
    TRAIN = 0
    TEST = 1


def read(fp):
    with open(fp,"r",newline="") as stream:
        reader = csv.reader(stream,delimiter=";")
        next(reader)            # ignore header
        for row in reader:
            entry = Entry(*row)
            yield entry

def add_to_global(
        entity:str,
        entity_md:List,
        entity_idx_mapping:Dict[str,Optional[int]],
        id_,
        **kwargs):
    if entity in entity_idx_mapping:
        return
    entity_idx_mapping[entity] = None
    entity_dict = {
        "file_name" if entity.endswith("jpg") else "name":entity,
        "id":id_
    }
    entity_idx_mapping[entity] = len(entity_md)
    if kwargs:
        entity_dict = {**entity_dict,**kwargs}
    entity_md.append(entity_dict)
    id_ += 1
    return id_

def create_annotation(
        entry:Entry,
        cats,
        cats_idx_mapping,
        imgs,
        imgs_idx_mapping,
        anno_meta,
        id_):

    anno_dict = {}
    anno_dict["category_id"] = cats[cats_idx_mapping[entry.annotation_tag]]["id"]
    anno_dict["image_id"] = imgs[imgs_idx_mapping[entry.filename.split("/")[1]]]["id"]
    anno_dict["id"] = id_

    x = int(entry.upper_left_corner_x)
    y = int(entry.upper_left_corner_y)
    w = int(entry.lower_right_corner_x) - int(entry.upper_left_corner_x)
    h = int(entry.lower_right_corner_y) - int(entry.upper_left_corner_y)

    anno_dict["bbox"] = [x,y,w,h]
    anno_meta.append(anno_dict)

    id_ += 1
    return id_




def processAnnoFile(fp,ctrlFlow:FLOW):
    global IMAGE_COUNTER
    global CAT_COUNTER
    global ANN0_COUNTER

    entries = read(fp)
    for entry in entries:
        kwargs = {}
        file_name = entry.filename.split("/")[-1]
        cat = entry.annotation_tag
        kwargs["height"] = 960
        kwargs["width"] = 1280
        if ctrlFlow == FLOW.TRAIN:
            image_counter = add_to_global(
                file_name,
                TRAIN_IMAGES_META,
                TRAIN_IMAGES_IDX_MAPPING,
                IMAGE_COUNTER,**kwargs)

            cat_counter = add_to_global(cat,CATS_META,CATS_IDX_MAPPING,CAT_COUNTER)
            anno_counter = create_annotation(
                entry,
                CATS_META,
                CATS_IDX_MAPPING,
                TRAIN_IMAGES_META,
                TRAIN_IMAGES_IDX_MAPPING,
                TRAIN_ANNO_META,
                ANN0_COUNTER
            )
        else:
            image_counter = add_to_global(file_name,
                          TEST_IMAGES_META,
                          TEST_IMAGES_IDX_MAPPING,IMAGE_COUNTER,**kwargs)
            cat_counter = add_to_global(cat,CATS_META,CATS_IDX_MAPPING,CAT_COUNTER)
            anno_counter = create_annotation(
                entry,
                CATS_META,
                CATS_IDX_MAPPING,
                TEST_IMAGES_META,
                TEST_IMAGES_IDX_MAPPING,
                TEST_ANNO_META,
                ANN0_COUNTER
            )

        IMAGE_COUNTER = image_counter or IMAGE_COUNTER
        CAT_COUNTER = cat_counter or CAT_COUNTER
        ANN0_COUNTER = anno_counter or ANN0_COUNTER

def processAllAnnoFiles(path:str):
    for file_path in os.listdir(path):
        realpath = os.path.join(path,file_path)
        if os.path.isdir(realpath):
            processAllAnnoFiles(realpath)
        else:
            if "BOX" in realpath:
                ctrFlow = FLOW.TRAIN if any(x in realpath for x in TRAIN_VAL_SEQ) else FLOW.TEST
                processAnnoFile(realpath,ctrFlow)


def createMasterAnno(images_meta,annos):
    return {
        "categories": CATS_META,
        "images":images_meta,
        "annotations":annos
    }

def splitTrainVal(master_train):
    master_val = {}
    train_annos, val_annos = [], []

    train_imgs = random.sample(
        master_train["images"],int(TRAIN_SPLIT*len(master_train["images"]))
    )

    train_img_ids, train_img_names = zip(
        *((img_d["id"],img_d["file_name"]) for img_d in train_imgs)
    )


    train_img_ids, train_img_names = set(train_img_ids), set(train_img_names)

    val_imgs, val_img_names = zip(
        *((img_d,img_d["file_name"]) for img_d in master_train["images"]
         if img_d["id"] not in train_img_ids)
    )
    val_img_names = set(val_img_names)

    for anno in master_train["annotations"]:
        if anno["image_id"] in train_img_ids:
            train_annos.append(anno)
        else:
            val_annos.append(anno)


    master_train["images"] = train_imgs
    master_train["categories"] = CATS_META
    master_train["annotations"] = train_annos

    master_val["images"] = val_imgs
    master_val["categories"] = CATS_META
    master_val["annotations"] = val_annos

    return master_train, master_val, train_img_names, val_img_names

def recurse_path(path:str, results):
    for fp in os.listdir(path):
        realpath = os.path.join(path,fp)
        if os.path.isdir(realpath):
            recurse_path(realpath, results)
        else:
            if realpath.endswith("jpg"):
                results.append(realpath)

    return results


def copyFilesToPath(train_img_ids,val_img_ids,test_img_ids,train_fp,test_fp,val_fp,unknown_fp,all_image_paths):
    for path in all_image_paths:
        filename = os.path.basename(path)
        if filename in train_img_ids:
            shutil.copy(path,train_fp)
        elif filename in test_img_ids:
            shutil.copy(path,test_fp)
        elif filename in val_img_ids:
            shutil.copy(path,val_fp)
        else:
            shutil.copy(path,unknown_fp)

def dump_json(master,fp,name):
    realpath = os.path.join(fp,name)
    with open(realpath,"w") as stream:
        json.dump(master,stream)

def log_size(master,instance):
    print(f"Number of {instance} images = {len(master['images'])}")
    print(f"Number of {instance} categories = {len(master['categories'])}")
    print(f"Number of {instance} Annotations = {len(master['annotations'])}")




def main(anno_fp,img_fp,dump_fp):

    processAllAnnoFiles(anno_fp)
    master_train = createMasterAnno(TRAIN_IMAGES_META,TRAIN_ANNO_META)
    master_test = createMasterAnno(TEST_IMAGES_META,TEST_ANNO_META)

    master_train, master_val, train_img_names, val_img_names = splitTrainVal(master_train)

    print(f"Train Annotations: f{len(master_train)}")
    log_size(master_train,"train")
    log_size(master_val,"val")
    log_size(master_test,"test")

    all_image_paths = [recurse_path(os.path.join(img_fp,fp),[]) for fp in (TRAIN_VAL_SEQ+TEST_SEQ)]
    all_image_paths = [path for x in all_image_paths for path in x ] # flatten

    train_fp = os.path.join(dump_fp,"train")
    val_fp = os.path.join(dump_fp,"val")
    test_fp = os.path.join(dump_fp,"test")
    unknown_fp = os.path.join(dump_fp, "unknown")

    os.makedirs(train_fp,exist_ok=True)
    os.makedirs(val_fp,exist_ok=True)
    os.makedirs(test_fp,exist_ok=True)
    os.makedirs(unknown_fp,exist_ok=True)

    copyFunc = partial(copyFilesToPath, train_img_names,val_img_names,TEST_IMAGES_IDX_MAPPING,train_fp,test_fp,val_fp,unknown_fp)

    pool_size = multiprocessing.cpu_count() * 2
    splits = np.array_split(all_image_paths,100)

    # copyFilesToPath(train_img_ids, TEST_IMAGES_IDX_MAPPING, train_fp, test_fp,val_fp, splits[0])

    print("Lenghts of created splits")
    print(*[len(arr) for arr in splits[:10]],sep="/")

    pool = multiprocessing.Pool(processes=pool_size)
    pool.map(copyFunc,splits)
    pool.close()
    pool.join()

    dump_json(master_train,dump_fp,"train.json")
    dump_json(master_val,dump_fp,"val.json")
    dump_json(master_test,dump_fp,"test.json")



if __name__ == '__main__':
    import time
    start = time.time()
    main(ANNO_PATH,IMG_PATH,OUTPUT_PATH)
    print(f"Completed in {time.time()-start}s")
