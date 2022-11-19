# -*- coding: utf-8 -*-

import json
import os

def read_json(fp):
    with open(fp,"r") as stream:
        out = json.load(stream)
    return out

def test_annotations(annos,img_fps, img_mapping):
    found = 0
    not_found = 0
    total = 0
    key_errors = 0
    for anno in annos:
        try:
            if img_mapping[anno["image_id"]] in img_fps:
                found += 1
            else:
                not_found += 1
        except KeyError:
            key_errors += 1

        total += 1

    print(f"Total are {total}")
    print(f"found are {found}")
    print(f"not_found are {not_found}")
    print(f"key_errors are {key_errors}")



def main(json_fp,img_fp):
    annos = read_json(json_fp)
    img_fps = set(x for x in os.listdir(img_fp))
    img_mapping = {img["id"]:img["file_name"] for img in annos["images"]}
    test_annotations(annos["annotations"],img_fps , img_mapping)


# if __name__ == '__main__':
json_fp = "/home/pupil/projects/tf-detectron/datasets/lisa/test.json"
img_fp = "/home/pupil/projects/tf-detectron/datasets/lisa/test/"

# json_fp = "/home/pupil/projects/tf-detectron/datasets/lisa/val.json"
# img_fp = "/home/pupil/projects/tf-detectron/datasets/lisa/val/"

# json_fp = "/home/pupil/projects/tf-detectron/datasets/lisa/train.json"
# img_fp = "/home/pupil/projects/tf-detectron/datasets/lisa/train/"
main(json_fp,img_fp)
