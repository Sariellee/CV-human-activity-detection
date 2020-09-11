import os
import xml.etree.ElementTree as ET

import numpy
from cv2 import cv2

DATASET_PATH = './validation_data/'
FRAMES_PATH = DATASET_PATH + 'PICTURES/'
LABELS_PATH = DATASET_PATH + 'ANNOTATION/'


def read_labels(path: str) -> 'numpy.array':
    """Reads labels from labels folder and outputs them in a numpy array"""
    labels = []
    for label_path in os.listdir(path):
        tree = ET.parse(path + label_path)



def read_frames(path: str) -> 'numpy.array':
    """Reads frames from data folder and returns them in a numpy array"""
    frames = []
    for img_path in os.listdir(path):
        frames.append(cv2.imread(path + img_path))

    return frames


if __name__ == '__main__':
    frames = read_frames(FRAMES_PATH)
    labels = read_labels(LABELS_PATH)
