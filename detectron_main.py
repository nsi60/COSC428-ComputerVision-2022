# obtain
# detectron2
# 's default config
# self.cfg = get_cfg()
#
# # Load Model Config
# self.model = os.getenv('MODEL_CONFIG', 'mask_rcnn_R_50_FPN_3x.yaml')
# # load values from a file
# self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/" + self.model))
#
#
# # Additional Info when using cuda
# if torch.cuda.is_available():
#     self.cfg.MODEL.DEVICE = "cuda"
# else:
# # set device to cpu
#     self.cfg.MODEL.DEVICE = "cpu"
#
#
# # get Model from paperspace trained model directory
# model_path = os.path.abspath('/models/model/detectron/model_final.pth')
# if os.path.isfile(model_path):
#     print('Using Trained Model {}'.format(model_path), flush=True)
# else:
#     # Load default pretrained model from Model Zoo
#     print('No Model Found at {}'.format(model_path), flush=True)
#     model_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model)
#
# self.cfg.MODEL.WEIGHTS = model_path
#
#
# # detectron model
# def inference(self, file):
#
#     predictor = DefaultPredictor(self.cfg)
# 	im = cv.imread(file)
# 	rgb_image = im[:, :, ::-1]
# 	outputs = predictor(rgb_image)
#
# 	# get metadata
# 	metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
# 	# visualise
# 	v = Visualizer(rgb_image[:, :, ::-1], metadata=metadata, scale=1.2)
# 	v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#
# 	# get image
# 	img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))
#
# 	return img