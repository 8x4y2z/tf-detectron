# -*- coding: utf-8 -*-

import numpy as np
import cv2
import csv
from collections import namedtuple
import os

DATA_PATH = "/home/pupil/Documents/upgrad/msc/lisa"
CSV_PATH = "/home/pupil/Documents/upgrad/msc/lisa/Annotations/Annotations"
COLOR = (0,255,255)

LINE = cv2.LINE_AA
XOFFSET = YOFFSET = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
ANNO_COLOR = (255,255,100)
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

def viz(entry:Entry,img_path,dump_path,n):
    imgfp = img_path+f"/{entry.filename.split('/')[-1]}"
    arr = cv2.imread(imgfp)
    rect = cv2.rectangle(arr,(int(entry.upper_left_corner_x),int(entry.upper_left_corner_y)),
                         (int(entry.lower_right_corner_x),int(entry.lower_right_corner_y)),COLOR,
                         2)
    cv2.putText(arr,entry.annotation_tag,
                (int(entry.upper_left_corner_x)-XOFFSET,int(entry.upper_left_corner_y)-YOFFSET),
                FONT,0.4,ANNO_COLOR,1,LINE)
    sfp = dump_path+f"/{n}.jpg"
    cv2.imwrite(sfp,rect)



def read(fp,n,reverse):
    """fp: Full path to any of the csv file
    """

    # Read csv file
    with open(fp,"r",newline="") as csvfile:
        reader = csv.reader(csvfile,delimiter=";")
        if reverse:
            reader = reversed(list(reader))
        else:
            next(reader)      # ignore header
        for i,row in enumerate(reader):
            if i == n and n > 0:
                break
            entry = Entry(*row)
            yield entry

def main(annoLoc,saveLoc,ann_type="box",n=5,reverse=False):
    full_anno_path = CSV_PATH+f"/{annoLoc}/frameAnnotations{ann_type.upper()}.csv"
    img_path = (DATA_PATH +
                f"/{annoLoc.split('/')[0]}"
                f"/{annoLoc.split('/')[0]}"
                f"/{annoLoc.split('/')[1]}/frames")

    entries = read(full_anno_path,n,reverse)
    os.makedirs(saveLoc, exist_ok=True)
    for i,entry in enumerate(entries):
        viz(entry,img_path,saveLoc,i)


# if __name__ == '__main__':
annoLoc = "dayTrain/dayClip1/"
saveLoc = "/home/pupil/Documents/upgrad/msc/others/dump"
main(annoLoc,saveLoc,n=1000)
