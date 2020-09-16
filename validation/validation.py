import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
from cv2 import cv2
from pycocotools import mask as COCOmask

from src.model import MobileNet

DATASET_PATH = './validation/validation_data/'
FRAMES_PATH = DATASET_PATH + 'PICTURES/'
LABELS_PATH = DATASET_PATH + 'ANNOTATION/'


def read_labels(path: str) -> Dict[str, List[List[int]]]:
    """Reads labels from labels folder and outputs them in a numpy array"""
    labels = defaultdict(list)
    for label_path in os.listdir(path):
        tree = ET.parse(path + label_path)
        for obj in tree.iter('object'):
            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            labels[label_path[:-4]].append([xmin, ymin, xmax, ymax])
    return labels


def read_frames(path: str) -> Dict[str, 'np.array']:
    """Reads frames from data folder and returns them in a numpy array"""
    frames = {}
    for img_path in os.listdir(path):
        frames[img_path[:-4]] = (cv2.imread(path + img_path))

    return frames


def IoU(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def xyxy_to_xywh(xyxy: List[int]) -> Tuple[int, int, int, int]:
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    # Single box given as a list of coordinates
    assert len(xyxy) == 4
    x1, y1 = xyxy[0], xyxy[1]
    w = xyxy[2] - x1 + 1
    h = xyxy[3] - y1 + 1
    return x1, y1, w, h


def validate(model, frames: Dict[str, 'np.array'], labels: Dict[str, List[List[int]]]):
    sums = []
    model = model()
    frame_and_label = [(frames[key], labels[key]) for key in frames.keys()]
    for frame, label_bboxes in frame_and_label:
        predicted_bboxes, _ = model.predict(frame)
        bboxes = [xyxy_to_xywh(bbox) for bbox in predicted_bboxes]
        label = [xyxy_to_xywh(lbl) for lbl in label_bboxes]
        iou = COCOmask.iou(np.array(bboxes), np.array(label), [int(False)] * np.array(label).shape[0])

        if (label and not bboxes) or (not label and bboxes):
            score = 0
        elif not label and not bboxes:
            score = 1
        else:
            score = np.average([max(scores) for scores in iou])

            print(np.average([max(scores) for scores in iou]) if len(iou) else None)
        sums.append(score)

        # visual_boxes(frame, label_bboxes, predicted_bboxes)

    print(f"Average accuracy: {np.average(sums)}")


def measure_FPS(model, frames, labels):
    FPSes = []
    model = model()
    frame_and_label = [(frames[key], labels[key]) for key in frames.keys()]
    for frame, label_bboxes in frame_and_label:
        start = time.time()
        predicted_bboxes, _ = model.predict(frame)
        end = time.time()
        fps = 1/(end-start)
        FPSes.append(fps)
        print(f'FRS on frame: {fps}')

    print(np.average(FPSes))


def visual_boxes(frame, truth, predicted):
    """
    True labels are green.
    Predicted labels are red.
    """
    for bbox in predicted:
        xLeftBottom, yLeftBottom, xRightTop, yRightTop = bbox
        cv2.rectangle(
            frame,
            (xLeftBottom, yLeftBottom),
            (xRightTop, yRightTop),
            (0, 0, 255)
        )

    for bbox in truth:
        xLeftBottom, yLeftBottom, xRightTop, yRightTop = bbox
        cv2.rectangle(
            frame,
            (xLeftBottom, yLeftBottom),
            (xRightTop, yRightTop),
            (0, 255, 0)
        )

    cv2.imwrite('test.png', frame)


if __name__ == '__main__':
    frames = read_frames(FRAMES_PATH)
    labels = read_labels(LABELS_PATH)
    validate(MobileNet, frames, labels)
    # measure_FPS(MobileNet, frames, labels)
