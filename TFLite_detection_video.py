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
import os
import argparse
import time
import cv2
import numpy as np
import importlib.util
import random
from functions import *
from radar_broadcast import RadarBroadcast
from multiprocessing import Process, Value

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modeldir", help="Folder the .tflite file is located in", required=True)
parser.add_argument("--video", help="Name of the video file", default="test.mp4")
parser.add_argument(
    "--edgetpu", help="Use Coral Edge TPU Accelerator to speed up detection", action="store_true"
)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
VIDEO_NAME = args.video
use_TPU = args.edgetpu

# initialize radar


def broadcast_radar(dst, dng):
    ant_senddemo = RadarBroadcast(dst, dng)
    ant_senddemo.open_channel()


if __name__ == "__main__":
    # queue = Queue()
    dst = Value("d", 1.0)
    dng = Value("d", 0.0)
    p = Process(
        target=broadcast_radar,
        args=(
            dst,
            dng,
        ),
    )
    p.start()

    # ant_senddemo = RadarBroadcast(distance)
    # ant_senddemo.open_channel()
    # print(queue.get())    # prints "[42, None, 'hello']"
    # p.join()

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
    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)
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

    VIDEOS = []
    if os.path.isdir(VIDEO_NAME):
        # list all in dir
        for file in os.listdir(VIDEO_NAME):
            if file.endswith(".mp4"):
                VIDEOS.append(os.path.join(CWD_PATH, VIDEO_NAME, file))
        random.shuffle(VIDEOS)
    else:
        VIDEOS = [VIDEO_NAME]  # single video

    for video_path in VIDEOS:
        # Open video file
        video = cv2.VideoCapture(video_path)

        SCENE_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        SCENE_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_max_danger = 0.0

        while video.isOpened():
            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame = video.read()

            if not ret:
                print("Reached the end of the video!")
                break

            ############# VARIABLE SETTING #############

            # stop only when danger detected
            if frame_max_danger > DANGER_THRESHOLD or frame_max_danger is None:
                print(f"DNG: {frame_max_danger}")
                # input("")

            # reset values for frame
            frame_max_danger = 0.0
            frame_min_distance = 1.0

            ######### FRAME PREPROCESSING ###########

            input_data = preprocess_frame(height, width, frame, floating_model)

            ############# INFERENCE ###########
            boxes, classes, scores = inference(
                interpreter, input_data, input_details, output_details
            )

            ############# ROAD EDGE #############
            # get the road edge before we draw shapes into image
            road_edge = get_road_edge(frame)

            ############# SAFETY LINE #############
            cv2.line(
                frame,
                (get_safety_line_x(SCENE_HEIGHT, SCENE_WIDTH), SCENE_HEIGHT),
                (get_safety_line_x(0, SCENE_WIDTH), 0),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if (scores[i] > MIN_CONF_THRESHOLD) and (scores[i] <= 1.0):
                    # continue  # uncomment to disable boxes

                    ############# Get bounding box coordinates and draw box #############
                    ymin, xmin, ymax, xmax = get_box_coords(SCENE_WIDTH, SCENE_HEIGHT, boxes, i)

                    # original box of the detected cars (white 1px), before filtering/reshaping boxes
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

                    ########### 1. filter out if the box if the whole car is not detected (corners) ###########
                    if is_in_corner(SCENE_WIDTH, SCENE_HEIGHT, ymin, xmin, ymax, xmax):
                        continue

                    ########### 2. filter out if the box is smaller than the threshold ###########
                    if is_too_small(SCENE_WIDTH, SCENE_HEIGHT, ymin, xmin, ymax, xmax):
                        continue

                    ########### 3. filter out cars on the L side ###########
                    if is_left_side(SCENE_WIDTH, frame, ymin, xmin):
                        continue

                    ######## NEW BOX WIDTH (should be the cars front) ########
                    # original_box_width = xmax - xmin
                    # size_coeff = original_box_width * (0.45 * box_corner_dis)

                    # overtaking on R (camera view, driving on right side of the road)
                    # xmin = int(xmin + size_coeff)  # increase left corner of the box
                    # half_size = r_half_size

                    ######## A - distance from the center (left or right corner) ########

                    a = get_a(frame, xmin, ymax, SCENE_WIDTH)

                    ######## C - box proportions relative to scene size ########

                    # c = (ymax - ymin) / SCENE_HEIGHT
                    c, distance = get_c(ymin, xmin, ymax, xmax, SCENE_WIDTH, SCENE_HEIGHT)

                    ######## DANGER INDEX CALCULATION ########

                    # danger = (a * 0.33) + (b * 0.33) + (c * 0.33)
                    danger = a * c
                    # danger = c
                    # danger = a

                    danger = round(danger, 2)
                    color = (0, int((1 - danger) * 255), int(danger * 255))

                    # set the values per frame
                    frame_max_danger = danger if danger > frame_max_danger else frame_max_danger
                    frame_min_distance = (
                        distance if distance < frame_min_distance else frame_min_distance
                    )

                    ######## DRAWING BOXES ########

                    draw_detection(
                        labels,
                        frame,
                        classes,
                        scores,
                        i,
                        ymin,
                        xmin,
                        ymax,
                        xmax,
                        a,
                        c,
                        distance,
                        danger,
                        color,
                    )

            # draw last # BL points
            for p in POINTS[-DRAW_N_POINTS:]:
                cv2.circle(frame, p, radius=0, color=(0, 255, 0), thickness=3)

            # draw road edge (single line for whole frame)
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

            # queue.put([distance, frame_max_danger])

            dst.value = frame_min_distance
            dng.value = frame_max_danger

            # input("")  # ENTER to move to next frame
            time.sleep(0.0005)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord("q"):
                break

        # Clean up
        video.release()
        cv2.destroyAllWindows()
