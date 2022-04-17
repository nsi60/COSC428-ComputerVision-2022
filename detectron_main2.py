import math
import os
import pickle
import time
from threading import Thread

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_checkpoint_url
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask

import pyrealsense2 as rs
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Detector:
    def __init__(self):
        cfg_save_path = "custom_model/IS_cfg.pickle"

        with open(cfg_save_path, 'rb') as f:
            self.cfg = pickle.load(f)

        # self.cfg = get_cfg()
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/custom_mask_rcnn_R_50_FPN_3x.yaml"))
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/custom_mask_rcnn_R_50_FPN_3x.yaml")
        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 #or 0.5

        self.cfg.MODEL.WEIGHTS = os.path.join("instance_segmentation/", "model_final.pth")  # cfg.OUTPUT_DIR
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        viz = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    def onVideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if (cap.isOpened() == False):
            print("Error opening file...")
            return

        (success, image) = cap.read()

        while success:
            predictions = self.predictor(image)
            v = Visualizer(image[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
            output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

            cv2.imshow("Result", output.get_image()[:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break



# detector = Detector()
#
# image_path1 = "desk_dataset/images/08287504-Starmore-office-desk-angle_1024x1024.webp"
# image_path2 = "desk_dataset/images/0ac6b70d-images234.jpeg"
# image_path3 = "desk_dataset/images/0f5db8f8-Agile-Winder-Desk-McGreals-11289__462629_Detail_Listing_Standard_DesktopW10.jpg"
# image_path4 = "desk_dataset/images/1a10aba8-view.jpeg"
# image_path5 = "desk_dataset/images/2e71dabf-enhance-black-oak_384x384.webp"
# image_path6 = "desk_dataset/images/f5fe3565-images4564.jpeg"
#
# image_path_list = [image_path1,image_path2,image_path3,image_path4,image_path5,image_path6]
#
# video_path = "desk_dataset/work_vid.mp4"
#
# # for img in image_path_list:
# #   detector.onImage(img)
#
#
# detector.onVideo(video_path)


# Resolution of camera streams
RESOLUTION_X = 640  # 640, 1280
RESOLUTION_Y = 480  # 360(BW:cannot work in this PC, min:480)  #480, 720

# Configuration for histogram for depth image
NUM_BINS = 500  # 500 x depth_scale = e.g. 500x0.001m=50cm
MAX_RANGE = 10000  # 10000xdepth_scale = e.g. 10000x0.001m=10m

AXES_SIZE = 10

# Set test score threshold
SCORE_THRESHOLD = 0.65  # vip-The smaller, the faster.


class VideoStreamer:
    """
    Video streamer that takes advantage of multi-threading, and continuously is reading frames.
    Frames are then ready to read when program requires.
    """

    def __init__(self, video_file=None):
        """
        When initialised, VideoStreamer object should be reading frames
        """
        self.setup_image_config(video_file)
        self.configure_streams()
        self.stopped = False

    def start(self):
        """
        Initialise thread, update method will run under thread
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """
        Constantly read frames until stop() method is introduced
        """
        while True:

            if self.stopped:
                return

            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # Convert image to numpy array and initialise images
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())

    def stop(self):
        self.pipeline.stop()
        self.stopped = True

    def read(self):
        return (self.color_image, self.depth_image)

    def setup_image_config(self, video_file=None):
        """
        Setup config and video steams. If --file is specified as an argument, setup
        stream from file. The input of --file is a .bag file in the bag_files folder.
        .bag files can be created using d435_to_file in the tools folder.
        video_file is by default None, and thus will by default stream from the
        device connected to the USB.
        """
        config = rs.config()

        if video_file is None:

            config.enable_stream(rs.stream.depth, RESOLUTION_X, RESOLUTION_Y, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, RESOLUTION_X, RESOLUTION_Y, rs.format.bgr8, 30)
        else:
            try:
                config.enable_device_from_file("bag_files/{}".format(video_file))
            except:
                print("Cannot enable device from: '{}'".format(video_file))

        self.config = config

    def configure_streams(self):
        # Configure video streams
        self.pipeline = rs.pipeline()

        # Start streaming
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

    def get_depth_scale(self):
        return self.profile.get_device().first_depth_sensor().get_depth_scale()


class Predictor(DefaultPredictor):
    def __init__(self):
        self.config = self.setup_predictor_config()
        super().__init__(self.config)

    def create_outputs(self, color_image):
        self.outputs = self(color_image)

    def setup_predictor_config(self):
        # cfg_file = get_config_file(config_path)

        # cfg = get_cfg()
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # pretrained = True
        # if pretrained:
        #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        #         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        #
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
        # # Mask R-CNN ResNet101 FPN weights
        # ##cfg.MODEL.WEIGHTS = "model_final_a3ec72.pkl"  #Load local model
        # # This determines the resizing of the image. At 0, resizing is disabled.
        # cfg.INPUT.MIN_SIZE_TEST = 0
        #
        # return cfg

        cfg_save_path = "custom_model/IS_cfg.pickle"

        with open(cfg_save_path, 'rb') as f:
            cfg = pickle.load(f)

        cfg.MODEL.WEIGHTS = os.path.join("custom_model/instance_segmentation/", "model_final.pth")  # cfg.OUTPUT_DIR
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        cfg.MODEL.DEVICE = "cuda"
        # Mask R-CNN ResNet101 FPN weights
        ##cfg.MODEL.WEIGHTS = "model_final_a3ec72.pkl"  #Load local model
        # This determines the resizing of the image. At 0, resizing is disabled.
        cfg.INPUT.MIN_SIZE_TEST = 0

        return cfg

    def format_results(self, class_names):
        """
        Format results so they can be used by overlay_instances function
        """
        predictions = self.outputs['instances']
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None

        labels = None
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        masks = predictions.pred_masks.cpu().numpy()
        masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]

        boxes_list = boxes.tensor.tolist()
        scores_list = scores.tolist()
        class_list = classes.tolist()

        for i in range(len(scores_list)):
            boxes_list[i].append(scores_list[i])
            boxes_list[i].append(class_list[i])

        boxes_list = np.array(boxes_list)

        return (masks, boxes, boxes_list, labels, scores_list, class_list)


class OptimizedVisualizer(Visualizer):
    """
    Detectron2's altered Visualizer class which converts boxes tensor to cpu
    """

    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.cpu().numpy()
        else:
            return np.asarray(boxes)


class DetectedObject:
    """
    Each object corresponds to all objects detected during the instance segmentation
    phase. Associated trackers, distance, position and velocity are stored as attributes
    of the object.
    masks[i], boxes[i], labels[i], scores_list[i], class_list[i]
    """

    def __init__(self, mask, box, label, score, class_name):
        self.mask = mask
        self.box = box
        self.label = label
        self.score = score
        self.class_name = class_name

    # BW: comment below for speed-up! ~5sec/frame faster.
    # def __str__(self):
    #     ret_str = "The pixel mask of {} represents a {} and is {}m away from the camera.\n".format(self.mask, self.class_name, self.distance)
    #     if hasattr(self, 'track'):
    #         if hasattr(self.track, 'speed'):
    #             if self.track.speed >= 0:
    #                 ret_str += "The {} is travelling {}m/s towards the camera\n".format(self.class_name, self.track.speed)
    #             else:
    #                 ret_str += "The {} is travelling {}m/s away from the camera\n".format(self.class_name, abs(self.track.speed))
    #         if hasattr(self.track, 'impact_time'):
    #             ret_str += "The {} will collide in {} seconds\n".format(self.class_name, self.track.impact_time)
    #         if hasattr(self.track, 'velocity'):
    #             ret_str += "The {} is located at {} and travelling at {}m/s\n".format(self.class_name, self.track.position, self.track.velocity)
    #     return ret_str

    def create_vector_arrow(self):
        """
        Creates direction arrow which will use Arrow3D object. Converts vector to a suitable size so that the direction is clear.
        NOTE: The magnitude of the velocity is not represented through this arrow. The arrow lengths are almost all identical
        """
        arrow_ratio = AXES_SIZE / max(abs(self.track.velocity_vector[0]), abs(self.track.velocity_vector[1]),
                                      abs(self.track.velocity_vector[2]))
        self.track.v_points = [x * arrow_ratio for x in self.track.velocity_vector]


class Arrow3D(FancyArrowPatch):
    """
    Arrow used to demonstrate direction of travel for each object
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def find_mask_centre(mask, color_image):
    """
    Finding centre of mask using moments
    """
    moments = cv2.moments(np.float32(mask))

    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    return cX, cY


def find_median_depth(mask_area, num_median, histg):
    """
    Iterate through all histogram bins and stop at the median value. This is the
    median depth of the mask.
    """

    median_counter = 0
    centre_depth = "0.00"
    for x in range(0, len(histg)):
        median_counter += histg[x][0]
        if median_counter >= num_median:
            # Half of histogram is iterated through,
            # Therefore this bin contains the median
            centre_depth = x / 50
            break

    return float(centre_depth)


def debug_plots(color_image, depth_image, mask, histg, depth_colormap):
    """
    This function is used for debugging purposes. This plots the depth color-
    map, mask, mask and depth color-map bitwise_and, and histogram distrobutions
    of the full image and the masked image.
    """
    full_hist = cv2.calcHist([depth_image], [0], None, [NUM_BINS], [0, MAX_RANGE])
    masked_depth_image = cv2.bitwise_and(depth_colormap, depth_colormap, mask=mask)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.imshow(depth_colormap)

    plt.subplot(2, 2, 2)
    plt.imshow(masks[i].mask)

    plt.subplot(2, 2, 3).set_title(labels[i])
    plt.imshow(masked_depth_image)

    plt.subplot(2, 2, 4)
    plt.plot(full_hist)
    plt.plot(histg)
    plt.xlim([0, 600])
    plt.show()


if __name__ == "__main__":
    # Initialise Detectron2 predictor
    predictor = Predictor()

    # Initialise video streams from D435
    video_streamer = VideoStreamer()

    # Initialise Kalman filter tracker from modified Sort module
    # mot_tracker = Sort()

    depth_scale = video_streamer.get_depth_scale()
    print("Depth Scale is: {:.4f}m".format(depth_scale))

    speed_time_start = time.time()

    video_streamer.start()
    time.sleep(1)

    while True:

        time_start = time.time()
        color_image, depth_image = video_streamer.read()
        detected_objects = []

        t1 = time.time()

        camera_time = t1 - time_start

        predictor.create_outputs(color_image)
        outputs = predictor.outputs

        t2 = time.time()
        model_time = t2 - t1
        print("Model took {:.2f} time".format(model_time))

        predictions = outputs['instances']

        if outputs['instances'].has('pred_masks'):
            num_masks = len(predictions.pred_masks)
        # else:
        #     # Even if no masks are found, the trackers must still be updated
        #     tracked_objects = mot_tracker.update(boxes_list)
        #     continue

        detectron_time = time.time()

        # Create a new Visualizer object from Detectron2
        v = OptimizedVisualizer(color_image[:, :, ::-1], MetadataCatalog.get(predictor.config.DATASETS.TRAIN[0]))

        masks, boxes, boxes_list, labels, scores_list, class_list = predictor.format_results(
            v.metadata.get("thing_classes"))

        for i in range(num_masks):
            try:
                detected_obj = DetectedObject(masks[i], boxes[i], labels[i], scores_list[i], class_list[i])
            except:
                print("Object doesn't meet all parameters")

            detected_objects.append(detected_obj)

        # tracked_objects = mot_tracker.update(boxes_list)

        v.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=None,
            assigned_colors=None,
            alpha=0.3
        )

        speed_time_end = time.time()
        total_speed_time = speed_time_end - speed_time_start
        speed_time_start = time.time()
        for i in range(num_masks):
            """
            Converting depth image to a histogram with num bins of NUM_BINS 
            and depth range of (0 - MAX_RANGE millimeters)
            """

            mask_area = detected_objects[i].mask.area()
            num_median = math.floor(mask_area / 2)

            histg = cv2.calcHist([depth_image], [0], detected_objects[i].mask.mask, [NUM_BINS], [0, MAX_RANGE])

            # Uncomment this to use the debugging function
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # debug_plots(color_image, depth_image, masks[i].mask, histg, depth_colormap)

            centre_depth = find_median_depth(mask_area, num_median, histg)
            detected_objects[i].distance = centre_depth
            cX, cY = find_mask_centre(detected_objects[i].mask._mask, v.output)

            v.draw_circle((cX, cY), (0, 0, 0))
            v.draw_text("{:.2f}m".format(centre_depth), (cX, cY + 20))

            # for i in detected_objects:
            # print(i)

            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # cv2.imshow('Segmented Image', color_image)
        cv2.imshow('Segmented Image', v.output.get_image()[:, :, ::-1])
        # cv2.imshow('Depth', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time_end = time.time()
        total_time = time_end - time_start

        print("Time to process frame: {:.2f}".format(total_time))
        print("FPS: {:.2f}\n".format(1 / total_time))

    video_streamer.stop()
    cv2.destroyAllWindows()











