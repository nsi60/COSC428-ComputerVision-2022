from os import listdir
from os.path import isfile, join
import cv2
import click

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


@click.command()
@click.argument("dataset-name")
@click.argument("image-dir")
@click.argument("model-path")
def run(dataset_name, image_dir, model_path):
    MetadataCatalog.get(dataset_name).thing_classes = ["MTG", "Pokemon", "Yugioh"]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_path # path to the model we just trained
    predictor = DefaultPredictor(cfg)

    images = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f)) and f.endswith(".png")]
    for image_path in images:
        im = cv2.imread(image_path)
        outputs = predictor(im)
            
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(
            im[:, :, ::-1],
            scale=0.5, 
            metadata=MetadataCatalog.get(dataset_name)
        )
        out = v.draw_instance_predictions(outputs["instances"][0].to("cpu"))
        cv2.imshow("Detections", out.get_image()[:, :, ::-1])
 
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    run()
