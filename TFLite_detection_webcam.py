######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description:
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from functions import *


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modeldir", help="Folder the .tflite file is located in", required=True)
parser.add_argument(
    "--edgetpu", help="Use Coral Edge TPU Accelerator to speed up detection", action="store_true"
)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
SCENE_WIDTH = 1280
SCENE_HEIGHT = 720
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

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

DEBUG_DRAW = False

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

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# TODO: setup picamera before stream starts

# Initialize video stream
videostream = VideoStream(resolution=(SCENE_WIDTH, SCENE_HEIGHT), framerate=30).start()
time.sleep(1)

# for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()
    frame = frame1.copy()

    ############# VARIABLE SETTING #############
    frame_max_danger = 0.0  # reset danger for frame

    ############# INFERENCE ###########
    input_data = preprocess_frame(height, width, frame, floating_model)
    boxes, classes, scores = inference(interpreter, input_data, input_details, output_details)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if (scores[i] > MIN_CONF_THRESHOLD) and (scores[i] <= 1.0):
            ############# Get bounding box coordinates and draw box #############
            ymin, xmin, ymax, xmax = get_box_coords(SCENE_WIDTH, SCENE_HEIGHT, boxes, i)

            # original box of the detected cars (white 1px), before filtering/reshaping boxes
            if DEBUG_DRAW:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

            ########### 1. filter out if the box if the whole car is not detected (corners) ###########
            if is_in_corner(SCENE_WIDTH, SCENE_HEIGHT, ymin, xmin, ymax, xmax):
                continue

            ########### 2. filter out if the box is smaller than the threshold ###########
            if is_too_small(SCENE_WIDTH, SCENE_HEIGHT, ymin, xmin, ymax, xmax):
                continue

            ########### 3. filter out cars on the L side ###########
            if is_left_side(SCENE_WIDTH, frame, ymin, xmin, draw=DEBUG_DRAW):
                continue

            ######## A - distance from the center (left or right corner) ########

            a = get_a(frame, xmin, ymax, SCENE_WIDTH, draw=DEBUG_DRAW)

            ######## C - box proportions relative to scene size ########

            # c = (ymax - ymin) / SCENE_HEIGHT
            c, distance = get_c(ymin, xmin, ymax, xmax, SCENE_WIDTH, SCENE_HEIGHT)

            ######## DANGER INDEX CALCULATION ########

            danger = a * c

            danger = round(danger, 2)
            color = (0, int((1 - danger) * 255), int(danger * 255))
            frame_max_danger = danger if danger > frame_max_danger else frame_max_danger

            if DEBUG_DRAW:
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

    if frame_max_danger > DANGER_THRESHOLD:
        print(f"DANGER: {frame_max_danger}")

    # Draw framerate in corner of frame
    if DEBUG_DRAW:
        cv2.putText(
            frame,
            "FPS: {0:.2f}".format(frame_rate_calc),
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # All the results have been drawn on the frame, so it's time to display it.
        if DEBUG_DRAW:
            cv2.imshow("Object detector", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord("q"):
            break

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1


# Clean up
cv2.destroyAllWindows()
videostream.stop()
