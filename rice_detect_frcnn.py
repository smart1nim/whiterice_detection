import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import cv2
import pandas as pd

import matplotlib
matplotlib.use("TkAgg") #added as fix during visualisation
import matplotlib.pyplot as plt

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data import detection_utils as utils
import copy
def custom_mapper(input_dict):
    dataset_dict = copy.deepcopy(input_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [T.Resize((1200,1200)),
                      T.RandomFlip(prob=0.6, horizontal=True, vertical=False),
                      T.RandomFlip(prob=0.6, horizontal=False, vertical=True),
                      T.RandomContrast(0.7, 3.2),
                      T.RandomBrightness(0.6, 1.8),
                      ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

# write a function that loads the dataset into detectron2's standard format
def get_whiterice_dicts(csv_file, img_dir):
    df = pd.read_csv(csv_file)
    df['filename'] = df['filename'].map(lambda x: img_dir + x)

    classes = ['rice']

    df['class_int'] = df['class'].map(lambda x: classes.index(x))

    dataset_dicts = []
    for filename in df['filename'].unique().tolist():
        record = {}

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width

        objs = []
        for index, row in df[(df['filename'] == filename)].iterrows():
            obj = {
                'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': row['class_int'],
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return (dataset_dicts)



import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

classes = ['rice']

for d in ["train", "test"]:
    DatasetCatalog.register('whiterice/' + d, lambda d=d: get_whiterice_dicts('whiterice/' + d + '_labels.csv', 'whiterice/' + d+'/'))
    MetadataCatalog.get('whiterice/' + d).set(thing_classes=classes)


microcontroller_metadata = MetadataCatalog.get('whiterice/train')
'''
dataset_dicts = get_microcontroller_dicts('whiterice/train_labels.csv', 'whiterice/train/')
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    v = Visualizer(img[:, :, ::-1], metadata = microcontroller_metadata, scale=0.5)
    v = v.draw_dataset_dict(d)
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()

'''

#def do_training():
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ('whiterice/train',)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

#if __name__ == "__main__":
#   cfg = get_cfg()

#do_training(cfg)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
cfg.DATASETS.TEST = ('whiterice/test', )
predictor = DefaultPredictor(cfg)

df_test = pd.read_csv('whiterice/test_labels.csv')

import os, random

aa = os.listdir("whiterice/validate")
print(aa,"@@@")


for aa in os.listdir("whiterice/validate"):
#dataset_dicts = DatasetCatalog.get('whiterice/test')
#for d in random.sample(dataset_dicts, 5):
    temppath = os.path.join(os.getcwd(),"whiterice/validate")
    aa = os.path.join(temppath,aa)

    im = cv2.imread(aa)
    print(im)
    outputs = predictor(im)
    print(outputs)
    print(type(outputs["instances"]))
    print(outputs["instances"].pred_classes)
    if len(outputs["instances"].pred_classes):
        print("Yes Rice is present")

    v = Visualizer(im[:, :, ::-1], metadata=microcontroller_metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()

