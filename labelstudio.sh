#!/bin/bash

if [ ! -d "labelstudio-env" ]
then
    python3 -m venv "labelstudio-env"  # "env" is the name of the environment here.
    source "labelstudio-env/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install pycocotools
    python -m pip install Cython
    python -m pip install label-studio
    python -m pip install torch torchvision
    python -m pip install label-studio-ml
    python -m pip install jinja==3.0.3
    python -m pip install flask==2.0.3
else
    source "labelstudio-env/bin/activate"
fi

if [ ! -d "labelstudio" ]
then
    echo "Can't find the labelstudio directory. Are you inside the lab4 directory?"
    exit 1
else
    cd "labelstudio"
    if [ -d "labelstudio" ]
    then
        label-studio start labelstudio
    else    
        label-studio start labelstudio --init
    fi
fi