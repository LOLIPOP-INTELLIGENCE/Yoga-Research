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

img_path = 'CorrectPoses/warrior.png'
image = tf.io.read_file(img_path)
image = tf.io.decode_jpeg(image)
print(image.shape)
person = detect(image)

# print(person)
print(person.keypoints)

pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]for keypoint in person.keypoints],dtype=np.float32)

coordinates = pose_landmarks.flatten().astype(str).tolist()

print(coordinates)


with open('yoga_image.csv', 'w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(coordinates)

# Load the CSV file
# data_path = 'yoga_poses_5/yoga_test_data.csv'
data_path = 'yoga_image.csv'
data = pd.read_csv(data_path, header=None)

# Assuming we're visualizing the first row for demonstration
print(data)
row = data.iloc[0]

print(row)

# Load the image
# image_path = f'yoga_poses_5/test/{row.file_name}'
print(img_path)

image = cv2.imread(img_path)

# Define a list of keypoints in the order they appear in the CSV
keypoints = [
    ('nose_x', 'nose_y'),
    ('left_eye_x', 'left_eye_y'),
    ('right_eye_x', 'right_eye_y'),
    ('left_ear_x', 'left_ear_y'),
    ('right_ear_x', 'right_ear_y'),
    ('left_shoulder_x', 'left_shoulder_y'),
    ('right_shoulder_x', 'right_shoulder_y'),
    ('left_elbow_x', 'left_elbow_y'),
    ('right_elbow_x', 'right_elbow_y'),
    ('left_wrist_x', 'left_wrist_y'),
    ('right_wrist_x', 'right_wrist_y'),
    ('left_hip_x', 'left_hip_y'),
    ('right_hip_x', 'right_hip_y'),
    ('left_knee_x', 'left_knee_y'),
    ('right_knee_x', 'right_knee_y'),
    ('left_ankle_x', 'left_ankle_y'),
    ('right_ankle_x', 'right_ankle_y')
]

# Define connections between keypoints to draw the skeleton
skeleton = [
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle')
]

import math

def calculate_angle(A, B, C):
    # Adjust points for the inverted y-axis: invert the y-component
    BA = (A[0] - B[0], B[1] - A[1])  # Invert y-component
    BC = (C[0] - B[0], B[1] - C[1])  # Invert y-component
    
    # Dot product of AB and BC
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    
    # Magnitude of AB and BC
    magnitude_AB = math.sqrt(BA[0]**2 + BA[1]**2)
    magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)
    
    # Cosine of the angle
    cos_theta = dot_product / (magnitude_AB * magnitude_BC)
    
    # Clamp cos_theta to the interval [-1, 1] to avoid math domain error
    cos_theta = max(min(cos_theta, 1), -1)
    
    # Angle in radians
    angle_radians = math.acos(cos_theta)
    
    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees


right_hip = 12
right_knee = 14
right_ankle = 16

right_hip_point = (int(row[right_hip * 3]), int(row[right_hip * 3 + 1]))
right_knee_point = (int(row[right_knee * 3]), int(row[right_knee * 3 + 1]))
right_ankle_point = (int(row[right_ankle * 3]), int(row[right_ankle * 3 + 1]))

print(right_hip_point)
print(right_knee_point)
print(right_ankle_point)

# Calculate the angle between the right hip, right knee, and right ankle
angle = calculate_angle(right_hip_point, right_knee_point, right_ankle_point)
print(f'Angle between right hip-right knee-right ankle: {angle:.2f} degrees')

# repeat for left side
left_hip = 11
left_knee = 13
left_ankle = 15

left_hip_point = (int(row[left_hip * 3]), int(row[left_hip * 3 + 1]))
left_knee_point = (int(row[left_knee * 3]), int(row[left_knee * 3 + 1]))
left_ankle_point = (int(row[left_ankle * 3]), int(row[left_ankle * 3 + 1]))

print(left_hip_point)
print(left_knee_point)
print(left_ankle_point)

# Calculate the angle between the left hip, left knee, and left ankle
angle = calculate_angle(left_hip_point, left_knee_point, left_ankle_point)
print(f'Angle between left hip-left knee-left ankle: {angle:.2f} degrees')

# repeat for knee, hip, shoulder

right_knee = 14
right_hip = 12
right_shoulder = 6

right_knee_point = (int(row[right_knee * 3]), int(row[right_knee * 3 + 1]))
right_hip_point = (int(row[right_hip * 3]), int(row[right_hip * 3 + 1]))
right_shoulder_point = (int(row[right_shoulder * 3]), int(row[right_shoulder * 3 + 1]))

print(right_knee_point)
print(right_hip_point)
print(right_shoulder_point)

# Calculate the angle between the right knee, right hip, and right shoulder
angle = calculate_angle(right_knee_point, right_hip_point, right_shoulder_point)
print(f'Angle between right knee-right hip-right shoulder: {angle:.2f} degrees')

# repeat for left side

left_knee = 13
left_hip = 11
left_shoulder = 5

left_knee_point = (int(row[left_knee * 3]), int(row[left_knee * 3 + 1]))
left_hip_point = (int(row[left_hip * 3]), int(row[left_hip * 3 + 1]))
left_shoulder_point = (int(row[left_shoulder * 3]), int(row[left_shoulder * 3 + 1]))

print(left_knee_point)
print(left_hip_point)
print(left_shoulder_point)

# Calculate the angle between the left knee, left hip, and left shoulder
angle = calculate_angle(left_knee_point, left_hip_point, left_shoulder_point)
print(f'Angle between left knee-left hip-left shoulder: {angle:.2f} degrees')

# repeat for hip, shoulder, elbow
right_hip = 12
right_shoulder = 6
right_elbow = 8

right_hip_point = (int(row[right_hip * 3]), int(row[right_hip * 3 + 1]))
right_shoulder_point = (int(row[right_shoulder * 3]), int(row[right_shoulder * 3 + 1]))
right_elbow_point = (int(row[right_elbow * 3]), int(row[right_elbow * 3 + 1]))

print(right_hip_point)
print(right_shoulder_point)
print(right_elbow_point)


# Calculate the angle between the right hip, right shoulder, and right elbow
angle = calculate_angle(right_hip_point, right_shoulder_point, right_elbow_point)
print(f'Angle between right hip-right shoulder-right elbow: {angle:.2f} degrees')

# repeat for left side
left_hip = 11
left_shoulder = 5
left_elbow = 7

left_hip_point = (int(row[left_hip * 3]), int(row[left_hip * 3 + 1]))
left_shoulder_point = (int(row[left_shoulder * 3]), int(row[left_shoulder * 3 + 1]))
left_elbow_point = (int(row[left_elbow * 3]), int(row[left_elbow * 3 + 1]))

print(left_hip_point)
print(left_shoulder_point)
print(left_elbow_point)

# Calculate the angle between the left hip, left shoulder, and left elbow
angle = calculate_angle(left_hip_point, left_shoulder_point, left_elbow_point)
print(f'Angle between left hip-left shoulder-left elbow: {angle:.2f} degrees')

# repeat for shoulder, elbow, wrist
right_shoulder = 6
right_elbow = 8
right_wrist = 10

right_shoulder_point = (int(row[right_shoulder * 3]), int(row[right_shoulder * 3 + 1]))
right_elbow_point = (int(row[right_elbow * 3]), int(row[right_elbow * 3 + 1]))
right_wrist_point = (int(row[right_wrist * 3]), int(row[right_wrist * 3 + 1]))

print(right_shoulder_point)
print(right_elbow_point)
print(right_wrist_point)

# Calculate the angle between the right shoulder, right elbow, and right wrist
angle = calculate_angle(right_shoulder_point, right_elbow_point, right_wrist_point)
print(f'Angle between right shoulder-right elbow-right wrist: {angle:.2f} degrees')

# repeat for left side
left_shoulder = 5
left_elbow = 7
left_wrist = 9

left_shoulder_point = (int(row[left_shoulder * 3]), int(row[left_shoulder * 3 + 1]))
left_elbow_point = (int(row[left_elbow * 3]), int(row[left_elbow * 3 + 1]))
left_wrist_point = (int(row[left_wrist * 3]), int(row[left_wrist * 3 + 1]))

print(left_shoulder_point)
print(left_elbow_point)
print(left_wrist_point)

# Calculate the angle between the left shoulder, left elbow, and left wrist
angle = calculate_angle(left_shoulder_point, left_elbow_point, left_wrist_point)
print(f'Angle between left shoulder-left elbow-left wrist: {angle:.2f} degrees')

# save all the angles
all_angles = {
    'right_hip_right_knee_right_ankle': calculate_angle(right_hip_point, right_knee_point, right_ankle_point),
    'left_hip_left_knee_left_ankle': calculate_angle(left_hip_point, left_knee_point, left_ankle_point),
    'right_knee_right_hip_right_shoulder': calculate_angle(right_knee_point, right_hip_point, right_shoulder_point),
    'left_knee_left_hip_left_shoulder': calculate_angle(left_knee_point, left_hip_point, left_shoulder_point),
    'right_hip_right_shoulder_right_elbow': calculate_angle(right_hip_point, right_shoulder_point, right_elbow_point),
    'left_hip_left_shoulder_left_elbow': calculate_angle(left_hip_point, left_shoulder_point, left_elbow_point),
    'right_shoulder_right_elbow_right_wrist': calculate_angle(right_shoulder_point, right_elbow_point, right_wrist_point),
    'left_shoulder_left_elbow_left_wrist': calculate_angle(left_shoulder_point, left_elbow_point, left_wrist_point)
}

print(all_angles)
# write to a json file
import json

with open('warrior.json', 'w') as file:
    json.dump(all_angles, file)



cnt = 0
# Draw keypoints
for point in keypoints:
    x = int(row[cnt])
    y = int(row[cnt + 1])
    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green color for keypoints
    cnt = cnt + 3

# Convert image from BGR to RGB for matplotlib display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show the image
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes
plt.show()