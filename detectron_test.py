#test.py


from detectron2.engine import DefaultPredictor

import os
import pickle

from detectron_utils import *

# cfg_save_path = "/content/drive/MyDrive/IS_cfg.pickle"

cfg_save_path = "custom_model/IS_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
  cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join("instance_segmentation/", "model_final.pth")   #cfg.OUTPUT_DIR
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

# image_path0 = "/content/drive/MyDrive/image_subset/more/home_1.jpg"
image_path1 = "desk_dataset/images/08287504-Starmore-office-desk-angle_1024x1024.webp"
image_path2 = "desk_dataset/images/0ac6b70d-images234.jpeg"
image_path3 = "desk_dataset/images/0f5db8f8-Agile-Winder-Desk-McGreals-11289__462629_Detail_Listing_Standard_DesktopW10.jpg"
image_path4 = "desk_dataset/images/1a10aba8-view.jpeg"
image_path5 = "desk_dataset/images/2e71dabf-enhance-black-oak_384x384.webp"
image_path6 = "desk_dataset/images/f5fe3565-images4564.jpeg"

video_path = "/content/drive/MyDrive/image_subset/more/work_vid.mp4"


#from utils.py
# image_path_list = [image_path0, image_path1,image_path2,image_path3,image_path4,image_path5]
image_path_list = [image_path1,image_path2,image_path3,image_path4,image_path5,image_path6]


for img in image_path_list:
  on_image(img, predictor)
# on_video(video_path, predictor)