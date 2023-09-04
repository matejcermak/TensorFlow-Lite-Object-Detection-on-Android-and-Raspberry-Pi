######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description:
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import math
import os
import argparse
import time
import cv2
import numpy as np
import sys
import importlib.util
import imutils

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modeldir", help="Folder the .tflite file is located in", required=True)
parser.add_argument(
    "--graph",
    help="Name of the .tflite file, if different than detect.tflite",
    default="detect.tflite",
)
parser.add_argument(
    "--labels",
    help="Name of the labelmap file, if different than labelmap.txt",
    default="labelmap.txt",
)
parser.add_argument(
    "--threshold", help="Minimum confidence threshold for displaying detected objects", default=0.5
)
parser.add_argument("--video", help="Name of the video file", default="test.mp4")
parser.add_argument(
    "--edgetpu", help="Use Coral Edge TPU Accelerator to speed up detection", action="store_true"
)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec("tflite_runtime")
if pkg:
    from tflite_runtime.interpreter import Interpreter

    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter

    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if GRAPH_NAME == "detect.tflite":
        GRAPH_NAME = "edgetpu.tflite"

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH, VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

ROAD_EDGE_TOP_X = 0.34
ROAD_EDGE_TOP_Y = 0.5
ROAD_EDGE_MIDDLE = 0.65

DANGER_THRESHOLD = 0.7  # immediate danger threshold > notify!


def canny_edges(frame):
    canny = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.GaussianBlur(canny, (5, 5), 0)
    return cv2.Canny(canny, 235, 250, apertureSize=5)


def filter_road_area(edges):
    # https://medium.com/swlh/lane-finding-with-computer-vision-techniques-bad24828dbc0
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    # Define a four sided polygon to mask
    imshape = edges.shape

    vertices = np.array(
        [
            [  #       X               Y
                (imshape[1] * 0.0, imshape[0]),  # bottom left
                (imshape[1] * ROAD_EDGE_TOP_X, imshape[0] * ROAD_EDGE_TOP_Y),  # top left
                (imshape[1] * ROAD_EDGE_MIDDLE, imshape[0] * ROAD_EDGE_TOP_Y),  # top right
                (imshape[1] * ROAD_EDGE_MIDDLE, imshape[0]),  # bottom right
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(edges, mask)


def get_road_edge(frame):
    # canny = imutils.auto_canny(frame)  # auto canny instead of preconfigured one
    canny = canny_edges(frame)
    canny_filtered = filter_road_area(canny)
    lines = cv2.HoughLines(
        image=canny_filtered,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        lines=None,
        min_theta=0.1,
        max_theta=0.78,
    )
    if lines is None:
        return None

    # thetas = [l[0][1] for l in lines]

    # selected_i = thetas.index(min(thetas))  # select the one with min theta (the most vertical line)
    selected_i = 0  # select first element (most voted)

    l = lines[selected_i][0]
    rho = l[0]
    theta = l[1]

    a = math.cos(theta)
    b = math.sin(theta)

    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

    return pt1, pt2


def get_safety_line_x(y):
    a = 1.35  # angle
    b = SCENE_WIDTH * 0.2  # offset
    # (y+b)/a     ==    y=(a*x)+b
    return int((y + b) / a)


# Load the label map
with open(PATH_TO_LABELS, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == "???":
    del labels[0]

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(
        model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate("libedgetpu.so.1.0")]
    )
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]

print(input_details)

floating_model = input_details[0]["dtype"] == np.float32

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]["name"]

if "StatefulPartitionedCall" in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)

imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

points = []

while video.isOpened():
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()

    if not ret:
        print("Reached the end of the video!")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])[
        0
    ]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]["index"])[
        0
    ]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]["index"])[
        0
    ]  # Confidence of detected objects

    ########### ROAD EDGE ##########
    # get the road edge before we draw shapes into image
    road_edge = get_road_edge(frame)

    ########### FOCAL CENTER ##########
    SCENE_WIDTH = int(imW)
    SCENE_HEIGHT = int(imH)
    focal_center = int(SCENE_WIDTH / 2) + 49  # TODO: find focal center automatically (40 px offset)

    r_half_size = SCENE_WIDTH - focal_center
    l_half_size = focal_center

    # draw center line
    # cv2.rectangle(frame, (focal_center, 0), (focal_center, SCENE_HEIGHT), (255, 255, 255), 3)

    # draw safety line
    # cv2.line(frame, (int(SCENE_WIDTH*.7),SCENE_HEIGHT), (int(SCENE_WIDTH*.3), 0), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(
        frame,
        (get_safety_line_x(SCENE_HEIGHT), SCENE_HEIGHT),
        (get_safety_line_x(0), 0),
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
            # continue  # uncomment to disable boxes

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # original box of the detected cars (white)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

            # skip the box if the whole car is not detected (corners)
            PADDING_ACCEPTED = 10
            if (
                ymin < PADDING_ACCEPTED
                or xmin < PADDING_ACCEPTED
                or ymax > (SCENE_HEIGHT - PADDING_ACCEPTED)
                or xmax > (SCENE_WIDTH - PADDING_ACCEPTED)
            ):
                continue

            # if the box is smaller than # of the screen - do not show
            if ((ymax - ymin) * (xmax - xmin)) / (SCENE_WIDTH * SCENE_HEIGHT) < 0.003:
                continue

            # r_dist_to_center = max(0, xmax - focal_center)
            # l_dist_to_center = max(0, focal_center - xmin)

            # distance of the box to the center of image (0-1) (L/R)
            # box_corner_dis = max(
            #     r_dist_to_center / r_half_size, l_dist_to_center / l_half_size
            # )  # ** 2

            # print(f"ldist:{l_dist_to_center / l_half_size} rdist:{r_dist_to_center / r_half_size} cornerdist:{box_corner_dis}")

            ######## NEW BOX WIDTH (should be the cars front) ########
            # original_box_width = xmax - xmin
            # size_coeff = original_box_width * (0.45 * box_corner_dis)

            # overtaking on R (camera view, driving on right side of the road)
            # xmin = int(xmin + size_coeff)  # increase left corner of the box
            corner = SCENE_WIDTH - xmin  # distance from R corner of scene to L corner of object
            # half_size = r_half_size

            ######## A - distance from the center (left or right corner) ########

            bottom_left_point = (xmin, ymax)
            safety_line_point = (get_safety_line_x(ymax), ymax)

            points.append(bottom_left_point)

            cv2.circle(frame, bottom_left_point, radius=0, color=(0, 255, 0), thickness=10)
            cv2.circle(frame, safety_line_point, radius=0, color=(0, 255, 0), thickness=10)

            # depends on whether we are overtaken from L/R
            # if box area overlapping the center fix on 1 (center_x in min)
            safety_line_dist = SCENE_WIDTH - get_safety_line_x(ymax)
            shorter_dst = min(corner, safety_line_dist)
            a = (shorter_dst / safety_line_dist) ** 3

            ######## C - box proportions relative to scene size ########
            # c = (ymax - ymin) / SCENE_HEIGHT
            # distance = (1 - ((ymax - ymin) * (xmax - xmin)) / (SCENE_WIDTH * SCENE_HEIGHT)) ** 2
            # box_size_relative = (ymax - ymin) / SCENE_HEIGHT
            box_size_relative = (xmax - xmin) / (SCENE_WIDTH)  # already recalculated width
            distance = (
                1 - box_size_relative
            ) ** 4  # TODO: trucks vs cars?? should only take front width

            # around # meters when the car starts to get dangerous (0% danger)
            DISTANCE_TRESHOLD = 1  # TODO: remove?
            c = max(0, (DISTANCE_TRESHOLD - distance)) / DISTANCE_TRESHOLD

            # weighted result (w_# must add up to 1)

            ######## DANGER INDEX CALCULATION ########

            # danger = (a * 0.33) + (b * 0.33) + (c * 0.33)
            danger = a * c
            # danger = c
            # danger = a

            danger = round(danger, 2)
            color = (0, int((1 - danger) * 255), int(danger * 255))

            # Draw label
            object_name = labels[
                int(classes[i])
            ]  # Look up object name from "labels" array using class index

            label = "{}:{:.0%} {:.0%} a:{:.2} c:{:.2} dst:{:.2}".format(
                object_name, scores[i], danger, a, c, distance
            )

            # draw a new recalculated box estimating car front only

            cv2.rectangle(
                frame, (xmin, ymin), (xmax, ymax), color, 7 if danger > DANGER_THRESHOLD else 2
            )

            FONT_SCALE = 0.5

            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 4
            )  # Get font size
            label_ymin = max(
                ymin, labelSize[1] + 10
            )  # Make sure not to draw label too close to top of window
            cv2.rectangle(
                frame,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                color,
                cv2.FILLED,
            )  # Draw white box to put label text in
            cv2.putText(
                frame,
                label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 0),
                1,
            )  # Draw label text

            # print(f'x={xmin}:{xmax}, y={ymin}:{ymax}')

            # h = ymax-ymin
            # w = xmax-xmin
            # print(f'w={w}, h={h}, h/w={h/w}')

    for p in points[-100:]:  # last # points
        cv2.circle(frame, p, radius=0, color=(0, 255, 0), thickness=3)

    # All the results have been drawn on the frame, so it's time to display it.

    input("")  # ENTER to move to next frame

    # draw road edge
    if road_edge is not None:
        cv2.line(frame, road_edge[0], road_edge[1], (0, 0, 255), 2, cv2.LINE_AA)

    # draw road edge region filter
    cv2.line(
        frame,
        (0, SCENE_HEIGHT),
        (int(SCENE_WIDTH * ROAD_EDGE_TOP_X), int(SCENE_HEIGHT * ROAD_EDGE_TOP_Y)),
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (int(SCENE_WIDTH * ROAD_EDGE_TOP_X), int(SCENE_HEIGHT * ROAD_EDGE_TOP_Y)),
        (int(SCENE_WIDTH * ROAD_EDGE_MIDDLE), int(SCENE_HEIGHT * ROAD_EDGE_TOP_Y)),
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (int(SCENE_WIDTH * ROAD_EDGE_MIDDLE), int(SCENE_HEIGHT * ROAD_EDGE_TOP_Y)),
        (int(SCENE_WIDTH * ROAD_EDGE_MIDDLE), int(SCENE_HEIGHT)),
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("Object detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
