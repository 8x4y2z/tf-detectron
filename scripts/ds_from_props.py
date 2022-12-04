# -*- coding: utf-8 -*-

import torch
import cv2
import json
import random
import os

random.seed(100)

PROPS_PATH = "/home/pupil/Documents/upgrad/msc/apm_output/inference/proposal_predictions.pth"
IMGS_PATH = "/home/pupil/Documents/upgrad/msc/datasets/lisa/train_val/"
VAL_IMGS_PATH = "/home/pupil/Documents/upgrad/msc/datasets/lisa/train_val_val"
TRAIN_ANNOS_PATH = "/home/pupil/Documents/upgrad/msc/datasets/lisa/train_val_true_train.json"
VAL_ANNOS_PATH = "/home/pupil/Documents/upgrad/msc/datasets/lisa/train_val_true_val.json"

NEW_TRAIN_IMG_PATH = "/home/pupil/Documents/upgrad/msc/datasets/lisa/train_props/"
NEW_VAL_IMG_PATH =  "/home/pupil/Documents/upgrad/msc/datasets/lisa/val_props/"
NEW_TRAIN_ANNO_FP = "/home/pupil/Documents/upgrad/msc/datasets/lisa/train_props.json"
NEW_VAL_ANNO_FP = "/home/pupil/Documents/upgrad/msc/datasets/lisa/val_props.json"

GIMG_ID = 0
GANO_ID = 0
def read_json(fp):
    with open(fp,"r") as stream:
        return json.load(stream)

def write_json(objects,fp):
    with open(fp,"w") as stream:
        json.dump(objects,stream)

def proposal_to_image(proposals,fn,img_annos,fp,anno_list,img_list,nfp):
    global GIMG_ID
    global GANO_ID
    for proposal in proposals:
        new_img = {}
        x0,y0,x1,y1 = tuple(int(i) for i in proposal)
        props_anns = []
        for anno in img_annos:
            xx0,yy0,w,h = anno["bbox"]
            if xx0>x0 and xx0+w<x1 and yy0>y0 and yy0+h<y1:
                props_anns.append(anno)
        arr = cv2.imread(fp+fn)
        cropped = arr[y0:y1,x0:x1]
        for anno in props_anns:
            new_anno = {}
            xx0,yy0,w,h = anno["bbox"]
            new_x0,new_y0 = xx0-x0,yy0-y0
            new_anno["image_id"] = GIMG_ID
            new_anno["id"] = GANO_ID
            new_anno["category_id"] = anno["category_id"]
            new_anno["bbox"] = [new_x0,new_y0,w,h]
            new_anno["iscrowd"] = 0
            anno_list.append(new_anno)
            GANO_ID+=1
        new_img["file_name"] = f"{GIMG_ID}.jpg"
        new_img["id"] = GIMG_ID
        new_img["height"] = y1-y0
        new_img["width"] = x1-x0
        img_list.append(new_img)
        cv2.imwrite(nfp+f"{GIMG_ID}.jpg",cropped)
        GIMG_ID += 1





def viz(labels,n,props=None):
    # import pdb; pdb.set_trace()
    toshow = random.sample(labels["annotations"],n)
    toshow_img_ids = [anno["image_id"] for anno in toshow]

    imgds = [imgd for imgd in labels["images"] if imgd["id"] in toshow_img_ids ]
    boxes = []
    if props is not None:
        for imgd in imgds:
            for i,id_ in enumerate(props["ids"]):
                if id_ == imgd["id"]:
                    boxes.append(props["boxes"][i])

    for anno in toshow:
        img_id = anno["image_id"]
        for i,imgd in enumerate(imgds):
            if imgd["id"] == img_id:
                img_fp = imgd["file_name"]
                break

        x,y,w,h = anno["bbox"]
        arr = cv2.imread((IMGS_PATH+img_fp))
        _ = cv2.rectangle(arr,(x,y),(x+w,y+h),(255,0,0),2)
        if props is not None:
            for bbox in boxes[i]:
                x1,y1,x2,y2 = tuple(int(a) for a in bbox)
                _ = cv2.rectangle(arr,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imwrite(("today"+img_fp),arr)


def main():
    train_labels = read_json(TRAIN_ANNOS_PATH)
    val_labels = read_json(VAL_ANNOS_PATH)
    props = torch.load(PROPS_PATH)
    # viz(train_labels,5,props)
    imgfn_mapping = {}
    imgann_mapping = {}
    NEW_TRAIN_IMGS = []
    NEW_TRAIN_ANNOS = []

    NEW_VAL_IMGS, NEW_VAL_ANNOS = [],[]
    for imgd_ in train_labels["images"]:
        imgfn_mapping[imgd_["id"]] = imgd_["file_name"],0

    for annod_ in train_labels["annotations"]:
        if annod_["image_id"] in imgann_mapping:
            imgann_mapping[annod_["image_id"]].append(annod_)
        else:
            imgann_mapping[annod_["image_id"]] = []

    for imgd_ in val_labels["images"]:
        imgfn_mapping[imgd_["id"]] = imgd_["file_name"],1

    for annod_ in val_labels["annotations"]:
        if annod_["image_id"] in imgann_mapping:
            imgann_mapping[annod_["image_id"]].append(annod_)
        else:
            imgann_mapping[annod_["image_id"]] = []

    os.makedirs(NEW_TRAIN_IMG_PATH,exist_ok=True)
    os.makedirs(NEW_VAL_IMG_PATH,exist_ok=True)

    for i, id_ in enumerate(props["ids"]):
        # if i> 5:
        #     break
        prop_boccs = props["boxes"][i]
        fn,grp_ = imgfn_mapping[id_]
        annos = imgann_mapping[id_]
        if grp_ < 1:
            anno_list = NEW_TRAIN_ANNOS
            img_list = NEW_TRAIN_IMGS
            nfp = NEW_TRAIN_IMG_PATH
        else:
            anno_list = NEW_VAL_ANNOS
            img_list = NEW_VAL_IMGS
            nfp = NEW_VAL_IMG_PATH

        proposal_to_image(prop_boccs,fn,annos,IMGS_PATH,anno_list,img_list,nfp)

    final_new_train_out = {
        "images":NEW_TRAIN_IMGS,
        "annotations":NEW_TRAIN_ANNOS,
        "categories":train_labels["categories"]
    }

    final_new_val_out = {
        "images":NEW_VAL_IMGS,
        "annotations":NEW_VAL_ANNOS,
        "categories":train_labels["categories"]
    }


    write_json(final_new_train_out,NEW_TRAIN_ANNO_FP)
    write_json(final_new_val_out,NEW_VAL_ANNO_FP)




if __name__ == '__main__':
    import time
    start = time.time()
    print("Started")
    main()
    print(f"Completed in {time.time() - start}s")
