import math
import cv2
import numpy as np

ROAD_EDGE_TOP_X = 0.0
ROAD_EDGE_TOP_Y = 0.2
ROAD_EDGE_MIDDLE = 0.65

SAFETY_LINE_ANGLE = 3  # 1.35
SAFETY_LINE_OFFSET = 1.4  # 0.45

DRAW_N_POINTS = 200

PADDING_ACCEPTED = 0  # set to zero to disable filtering

MIN_CONF_THRESHOLD = 0.5
DANGER_THRESHOLD = 0.7  # immediate danger threshold > notify!
DISTANCE_TRESHOLD = 1  # TODO: remove?

POINTS = []


def get_box_coords(SCENE_WIDTH, SCENE_HEIGHT, boxes, i):
    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
    ymin = int(max(1, (boxes[i][0] * SCENE_HEIGHT)))
    xmin = int(max(1, (boxes[i][1] * SCENE_WIDTH)))
    ymax = int(min(SCENE_HEIGHT, (boxes[i][2] * SCENE_HEIGHT)))
    xmax = int(min(SCENE_WIDTH, (boxes[i][3] * SCENE_WIDTH)))
    return ymin, xmin, ymax, xmax


def is_in_corner(SCENE_WIDTH, SCENE_HEIGHT, ymin, xmin, ymax, xmax):
    return (
        ymin < PADDING_ACCEPTED
        or xmin < PADDING_ACCEPTED
        or ymax > (SCENE_HEIGHT - PADDING_ACCEPTED)
        or xmax > (SCENE_WIDTH - PADDING_ACCEPTED)
    )


def is_too_small(SCENE_WIDTH, SCENE_HEIGHT, ymin, xmin, ymax, xmax):
    return (((ymax - ymin) * (xmax - xmin)) / (SCENE_WIDTH * SCENE_HEIGHT)) < 0.003


def is_left_side(SCENE_WIDTH, frame, ymin, xmin, draw=True):
    top_left_point = (xmin, ymin)
    top_left_safety_point = (get_safety_line_x(ymin, SCENE_WIDTH), ymin)

    # draw thick blue circles for detected distances from the safety line
    if draw:
        cv2.circle(frame, top_left_point, radius=0, color=(255, 0, 0), thickness=10)
        cv2.circle(frame, top_left_safety_point, radius=0, color=(255, 0, 0), thickness=10)

    is_left_side = top_left_point[0] < top_left_safety_point[0]
    return is_left_side


def draw_detection(
    labels, frame, classes, scores, i, ymin, xmin, ymax, xmax, a, c, distance, danger, color
):
    object_name = labels[
        int(classes[i])
    ]  # Look up object name from "labels" array using class index

    label = "{}:{:.0%} {:.0%} a:{:.2} c:{:.2} dst:{:.2}".format(
        object_name, scores[i], danger, a, c, distance
    )

    # draw a box rectangle for non-filtered shapes
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 7 if danger > DANGER_THRESHOLD else 2)

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
    )


def preprocess_frame(height, width, frame, floating_model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    return input_data


def inference(interpreter, input_deta, input_details, output_details):
    interpreter.set_tensor(input_details[0]["index"], input_deta)
    interpreter.invoke()

    boxes_idx, classes_idx, scores_idx = 1, 3, 0
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

    return boxes, classes, scores


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


def get_safety_line_x(y, SCENE_WIDTH):
    a = SAFETY_LINE_ANGLE  # angle
    b = SCENE_WIDTH * SAFETY_LINE_OFFSET  # offset
    # (y+b)/a     ==    y=(a*x)+b
    return int((y + b) / a)


def get_a(frame, xmin, ymax, SCENE_WIDTH, draw=True):
    bottom_left_point = (xmin, ymax)
    safety_line_point = (get_safety_line_x(ymax, SCENE_WIDTH), ymax)
    corner = SCENE_WIDTH - xmin  # distance from R corner of scene to L corner of object

    POINTS.append(bottom_left_point)

    # draw thick circles for detected distances
    if draw:
        cv2.circle(frame, bottom_left_point, radius=0, color=(0, 255, 0), thickness=10)
        cv2.circle(frame, safety_line_point, radius=0, color=(0, 255, 0), thickness=10)

    # if box area overlapping the center fix on 1 (center_x in min)

    # distance from safety line to R corner
    safety_line_dist = SCENE_WIDTH - get_safety_line_x(ymax, SCENE_WIDTH)
    shorter_dst = min(corner, safety_line_dist)

    return (shorter_dst / safety_line_dist) ** 3


def get_c(ymin, xmin, ymax, xmax, SCENE_WIDTH, SCENE_HEIGHT):
    box_size_relative = ((ymax - ymin) * (xmax - xmin)) / (SCENE_WIDTH * SCENE_HEIGHT)

    # box_size_relative = (ymax - ymin) / SCENE_HEIGHT
    # box_size_relative = (xmax - xmin) / (SCENE_WIDTH)  # already recalculated width
    # TODO: trucks vs cars?? should only take front width
    distance = (1 - box_size_relative) ** 16

    # around # meters when the car starts to get dangerous (0% danger)
    c = max(0, (DISTANCE_TRESHOLD - distance)) / DISTANCE_TRESHOLD
    return c, distance
