
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances

from detectron2.data import DatasetCatalog, MetadataCatalog

from custom_dataset_train import train_custom_dataset

"""Register data set"""

register_coco_instances("desk_dataset_polygon", {}, "desk_dataset_polygon/result.json", "desk_dataset_polygon")

import random
import cv2

from detectron2.utils.visualizer import Visualizer

dataset_dicts = DatasetCatalog.get("desk_dataset_polygon")
metadata = MetadataCatalog.get("desk_dataset_polygon")

print(dataset_dicts)

# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow(vis.get_image()[:, :, ::-1])

train_custom_dataset()