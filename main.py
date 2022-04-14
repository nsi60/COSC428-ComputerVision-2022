## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

#from yt
point = (0,0)
def show_distance(event, x, y, args, params):
    global point
    point = (x, y)


cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
# cv2.setMouseCallback('RealSense', show_distance)
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # from yt
        # cv2.circle(color_frame, point, 4, (0, 0, 255))
        # distance = depth_frame[point[1], point[0] - 20]

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))




        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # Standard OpenCV boilerplate for running the net:
        # height, width = color_image.shape[:2]
        # expected = 300
        # aspect = width / height
        # resized_image = cv2.resize(color_image, (round(expected * aspect), expected))
        # crop_start = round(expected * (aspect - 1) / 2)
        # crop_img = resized_image[0:expected, crop_start:crop_start + expected]
        #
        # net = cv2.dnn.readNetFromCaffe("../MobileNetSSD_deploy.prototxt", "../MobileNetSSD_deploy.caffemodel")
        # inScaleFactor = 0.007843
        # meanVal = 127.53
        # classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
        #               "bottle", "bus", "car", "cat", "chair",
        #               "cow", "diningtable", "dog", "horse",
        #               "motorbike", "person", "pottedplant",
        #               "sheep", "sofa", "train", "tvmonitor")
        #
        # blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
        # net.setInput(blob, "data")
        # detections = net.forward("detection_out")
        #
        # label = detections[0, 0, 0, 1]
        # conf = detections[0, 0, 0, 2]
        # xmin = detections[0, 0, 0, 3]
        # ymin = detections[0, 0, 0, 4]
        # xmax = detections[0, 0, 0, 5]
        # ymax = detections[0, 0, 0, 6]
        #
        # className = classNames[int(label)]
        #
        # cv2.rectangle(crop_img, (int(xmin * expected), int(ymin * expected)),
        #               (int(xmax * expected), int(ymax * expected)), (255, 255, 255), 2)
        # cv2.putText(crop_img, className,
        #             (int(xmin * expected), int(ymin * expected) - 5),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        #
        # cv2.imshow('RealSense', crop_img)

        cv2.imshow('RealSense', images)
        if cv2.waitKey(1) == ord('q'):
            break


finally:

    # Stop streaming
    pipeline.stop()



