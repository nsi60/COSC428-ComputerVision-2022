#train.py

from detectron2.utils.logger import setup_logger

from detectron_utils import get_train_cfg

setup_logger()
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle
#https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

output_dir = "instance_segmentation"
num_classes = 1

device = "cuda" #or cpu

train_dataset_name = "desk_dataset"
train_images = "image_subset/train"
train_json_anot_path = "image_subset/train.json"

test_dataset_name = "desk_dataset_test"
test_images = "image_subset/test"
test_json_anot_path = "image_subset/test.json"

cfg_save_path = "custom_model/IS_cfg.pickle"

#####
register_coco_instances(name=train_dataset_name, metadata={}, json_file=train_json_anot_path, image_root=train_images)

register_coco_instances(name=test_dataset_name, metadata={}, json_file=test_json_anot_path, image_root=test_images)


##import utils.py
# plot_sample(dataset_name=train_dataset_name, n=2)


######
def main():
  cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

  with open(cfg_save_path, 'wb') as f:
    pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

  trainer = DefaultTrainer(cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()



if __name__ == "__main__":
  main()