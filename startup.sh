#!/usr/bin/env bash

pip uninstall -y torch
apt-get update && apt-get install libgl1 -y
python -m venv proj
. proj/bin/activate
git clone https://github.com/8x4y2z/tf-detectron.git
cd tf-detectron
pip install -r requirements.txt
pip install pycocotools
pip install opencv-python
