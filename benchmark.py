import google_benchmark as benchmark
from google_benchmark import Counter

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
import torch
import torchvision
import cv2
import py_nms_simd
import numpy as np
import pis_nms
import mmcv.ops

from input_const import OD_IMAGE_FILES, OD_OUTPUT_PATHS, OD_IMAGE_PATHS

@benchmark.register
@benchmark.option.arg(0)
@benchmark.option.arg(1)
@benchmark.option.arg(2)
@benchmark.option.arg(3)
@benchmark.option.arg(4)
@benchmark.option.arg(5)
def nms_simd(state):
    boxes = np.load(OD_OUTPUT_PATHS[state.range(0)]["boxes"])
    scores = np.load(OD_OUTPUT_PATHS[state.range(0)]["scores"])

    box_indices = None
    while state:
        box_indices = py_nms_simd.run(boxes, scores, 0.6, 0.4)

    img = cv2.imread(OD_IMAGE_PATHS[state.range(0)].__str__())
    for box_idx in box_indices:
        cv2.rectangle(img, (int(boxes[box_idx, 0]), int(boxes[box_idx, 1])), (int(boxes[box_idx, 2]), int(boxes[box_idx, 3])), (0, 0, 1), 1)
    cv2.imwrite(f"{OD_IMAGE_FILES[state.range(0)].__str__()}_simd.jpg", img)


@benchmark.register
@benchmark.option.arg(0)
@benchmark.option.arg(1)
@benchmark.option.arg(2)
@benchmark.option.arg(3)
@benchmark.option.arg(4)
@benchmark.option.arg(5)
def tensorflow_nms(state):
    boxes = np.load(OD_OUTPUT_PATHS[state.range(0)]["boxes"])
    scores = np.load(OD_OUTPUT_PATHS[state.range(0)]["scores"])
    for i in range(boxes.shape[0]):
        boxes[i, [0, 1]] = boxes[i, [1, 0]]
        boxes[i, [2, 3]] = boxes[i, [3, 2]]
    while state:
        tf.image.non_max_suppression_with_scores(boxes, scores, 11, 0.4, 0.6)


@benchmark.register
@benchmark.option.arg(0)
@benchmark.option.arg(1)
@benchmark.option.arg(2)
@benchmark.option.arg(3)
@benchmark.option.arg(4)
@benchmark.option.arg(5)
def vision_nms_with_score_threshold(state):
    boxes_origin = torch.from_numpy(np.load(OD_OUTPUT_PATHS[state.range(0)]["boxes"]))
    scores_origin = torch.from_numpy(np.load(OD_OUTPUT_PATHS[state.range(0)]["scores"]))

    while state:
        state.pause_timing()
        boxes = boxes_origin.clone()
        scores = scores_origin.clone()
        state.resume_timing()

        candidates = scores > 0.6
        scores = scores[candidates]
        boxes = boxes[candidates]

        torchvision.ops.nms(boxes, scores, 0.4)

@benchmark.register
@benchmark.option.arg(0)
@benchmark.option.arg(1)
@benchmark.option.arg(2)
@benchmark.option.arg(3)
@benchmark.option.arg(4)
@benchmark.option.arg(5)
def vision_nms(state):
    boxes = torch.from_numpy(np.load(OD_OUTPUT_PATHS[state.range(0)]["boxes"]))
    scores = torch.from_numpy(np.load(OD_OUTPUT_PATHS[state.range(0)]["scores"]))

    while state:
        torchvision.ops.nms(boxes, scores, 0.4)


@benchmark.register
@benchmark.option.arg(0)
@benchmark.option.arg(1)
@benchmark.option.arg(2)
@benchmark.option.arg(3)
@benchmark.option.arg(4)
@benchmark.option.arg(5)
def faster_nms(state):
    boxes = np.load(OD_OUTPUT_PATHS[0]["boxes"])
    scores = np.load(OD_OUTPUT_PATHS[0]["scores"])

    while state:
        pis_nms.run(boxes, 0.4)

@benchmark.register
@benchmark.option.arg(0)
@benchmark.option.arg(1)
@benchmark.option.arg(2)
@benchmark.option.arg(3)
@benchmark.option.arg(4)
@benchmark.option.arg(5)
def faster_nms_with_scores_threshold_sort(state):
    boxes_origin = np.load(OD_OUTPUT_PATHS[0]["boxes"])
    scores_origin = np.load(OD_OUTPUT_PATHS[0]["scores"])

    while state:
        state.pause_timing()
        boxes = boxes_origin.copy()
        scores = scores_origin.copy()
        state.resume_timing()

        mask = scores > 0.6
        score_idx = scores.argsort()[mask]
        boxes = boxes[score_idx]

        pis_nms.run(boxes, 0.4)



@benchmark.register
@benchmark.option.arg(0)
@benchmark.option.arg(1)
@benchmark.option.arg(2)
@benchmark.option.arg(3)
@benchmark.option.arg(4)
@benchmark.option.arg(5)
def cv2_nms(state):
    boxes = np.load(OD_OUTPUT_PATHS[0]["boxes"])
    scores = np.load(OD_OUTPUT_PATHS[0]["scores"])
    for i in range(boxes.shape[0]):
        boxes[i, 2] -= boxes[i, 0]
        boxes[i, 3] -= boxes[i, 1]
    while state:
        cv2.dnn.NMSBoxes(boxes, scores, 0.6, 0.4)

@benchmark.register
@benchmark.option.arg(0)
@benchmark.option.arg(1)
@benchmark.option.arg(2)
@benchmark.option.arg(3)
@benchmark.option.arg(4)
@benchmark.option.arg(5)
def mmcv_nms(state):
    boxes = torch.from_numpy(np.load(OD_OUTPUT_PATHS[state.range(0)]["boxes"]))
    scores = torch.from_numpy(np.load(OD_OUTPUT_PATHS[state.range(0)]["scores"]))

    while state:
        mmcv.ops.nms(boxes, scores, 0.4, 0, 0.6)

if __name__ == "__main__":
    benchmark.main()