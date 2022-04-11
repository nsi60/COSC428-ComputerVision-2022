#!/bin/bash

# Install detectron2
if [ -z "$(pip3 list | grep detectron2)" ]
then
    # Depending on the version of CUDA/PyTorch, you may need to change this command.
    # There is a table in the Git repo outlining which version to use: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
    python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.6/index.html
fi

if [ ! -d detectron2 ]
then
    # Download the repo to access all of the predeveloped config files/scripts/etc.
    git clone https://github.com/facebookresearch/detectron2.git
fi

cd detectron2/demo

# # Run the model on the webcam.
python3 demo.py --webcam --confidence-threshold 0.95 --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --opts MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl

