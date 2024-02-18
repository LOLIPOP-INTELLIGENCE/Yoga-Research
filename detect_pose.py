import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)

# Load MoveNet Thunder model
import utils
from data import BodyPart
from ml import Movenet
movenet = Movenet('movenet_thunder')

# the input image to improve pose estimation accuracy.
def detect(input_tensor, inference_count=3):
  image_height, image_width, channel = input_tensor.shape
 
  movenet.detect(input_tensor.numpy(), reset_crop_region=True)
 
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor.numpy(), 
                            reset_crop_region=False)

  return person


image = tf.io.read_file('captured_photo.png')
image = tf.io.decode_jpeg(image)
person = detect(image)

print(person)
print(person.keypoints)

pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]for keypoint in person.keypoints],dtype=np.float32)

coordinates = pose_landmarks.flatten().astype(np.str).tolist()


with open('yoga_image.csv', 'w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(coordinates)

print(coordinates)

print(len(coordinates))