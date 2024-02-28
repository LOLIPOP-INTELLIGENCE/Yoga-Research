import csv
import itertools
import os
import sys
import tempfile
import tqdm
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = load_model('yoga_poses_5/weights.best.hdf5')

cap = cv2.VideoCapture(0)
classes = ['chair', 'cobra', 'dog', 'tree', 'warrior']


pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)

import utils
from data import BodyPart
from ml import Movenet
movenet = Movenet('movenet_thunder')

def detect(input_tensor, inference_count=3):
  image_height, image_width, channel = input_tensor.shape
 
  movenet.detect(input_tensor, reset_crop_region=True)
 
  for _ in range(inference_count - 1):
    # person = movenet.detect(input_tensor.numpy(), 
                            # reset_crop_region=False)
    person = movenet.detect(input_tensor, reset_crop_region=False)

  return person

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for Movenet
    # input_tensor = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
    # input_tensor = tf.cast(input_tensor, dtype=tf.int32)

    print(type(frame))
    # reshape from (1, 192, 192, 3) to (192, 192, 3)
    # input_tensor = tf.reshape(input_tensor, (192, 192, 3))

    # Detect pose
    person = detect(frame)

    print(person)

    # Assuming the detect function returns the person object correctly
    pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score] for keypoint in person.keypoints], dtype=np.float32)
    coordinates = pose_landmarks.flatten()

    if len(coordinates) == 51:  # Ensure the correct number of keypoints is detected
        x = coordinates.reshape(1, 51)
        
        # Classify pose
        prediction = model.predict(x)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display the classification result on the frame
        cv2.putText(frame, f'{predicted_class} ({confidence:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        print('Incorrect number of keypoints detected')
    
    # Display the frame
    cv2.imshow('Pose Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# image_path = 'test_classify.jpeg'
# image = tf.io.read_file(image_path)
# image = tf.io.decode_jpeg(image)
# person = detect(image)

# print(person)
# print(person.keypoints)

# pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]for keypoint in person.keypoints],dtype=np.float32)

# coordinates = pose_landmarks.flatten().astype(np.str).tolist()


# with open('yoga_image.csv', 'w', newline='') as file:

#     writer = csv.writer(file)
#     writer.writerow(coordinates)

# print(coordinates)

# print(len(coordinates))

# # Load the CSV file
# # data_path = 'yoga_poses_5/yoga_test_data.csv'
# data_path = 'yoga_image.csv'
# data = pd.read_csv(data_path, header=None)

# # Assuming we're visualizing the first row for demonstration
# print(data)
# row = data.iloc[0]

# print(row)

# # Load the image
# # image_path = f'yoga_poses_5/test/{row.file_name}'
# print(image_path)

# image = cv2.imread(image_path)

# # Define a list of keypoints in the order they appear in the CSV
# keypoints = [
#     ('nose_x', 'nose_y'),
#     ('left_eye_x', 'left_eye_y'),
#     ('right_eye_x', 'right_eye_y'),
#     ('left_ear_x', 'left_ear_y'),
#     ('right_ear_x', 'right_ear_y'),
#     ('left_shoulder_x', 'left_shoulder_y'),
#     ('right_shoulder_x', 'right_shoulder_y'),
#     ('left_elbow_x', 'left_elbow_y'),
#     ('right_elbow_x', 'right_elbow_y'),
#     ('left_wrist_x', 'left_wrist_y'),
#     ('right_wrist_x', 'right_wrist_y'),
#     ('left_hip_x', 'left_hip_y'),
#     ('right_hip_x', 'right_hip_y'),
#     ('left_knee_x', 'left_knee_y'),
#     ('right_knee_x', 'right_knee_y'),
#     ('left_ankle_x', 'left_ankle_y'),
#     ('right_ankle_x', 'right_ankle_y')
# ]

# # Define connections between keypoints to draw the skeleton
# skeleton = [
#     ('left_shoulder', 'left_elbow'),
#     ('left_elbow', 'left_wrist'),
#     ('right_shoulder', 'right_elbow'),
#     ('right_elbow', 'right_wrist'),
#     ('left_shoulder', 'right_shoulder'),
#     ('left_shoulder', 'left_hip'),
#     ('right_shoulder', 'right_hip'),
#     ('left_hip', 'right_hip'),
#     ('left_hip', 'left_knee'),
#     ('left_knee', 'left_ankle'),
#     ('right_hip', 'right_knee'),
#     ('right_knee', 'right_ankle')
# ]

# cnt = 0
# # Draw keypoints
# for point in keypoints:
#     x = int(row[cnt])
#     y = int(row[cnt + 1])
#     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green color for keypoints
#     cnt = cnt + 3

# # Convert image from BGR to RGB for matplotlib display
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Show the image
# plt.figure(figsize=(10, 10))
# plt.imshow(image_rgb)
# plt.axis('off')  # Hide axes
# plt.show()


# # # Step 1: Load the saved model
# model = load_model('yoga_poses_5/weights.best.hdf5')

# # Step 2: Prepare your input data (this is just a placeholder, adjust as necessary)
# # Example for a model expecting input shape of (None, 10)
# data_path = 'yoga_image.csv'
# data = pd.read_csv(data_path, header=None)

# # Assuming we're visualizing the first row for demonstration
# print(data)
# row = data.iloc[0]

# print(row)

# x = np.array(row)

# x = x.reshape(1, 51)

# print(x)



# # # Step 3: Make a prediction
# prediction = model.predict(x)

# classes = ['chair', 'cobra', 'dog', 'tree', 'warrior']

# print("Prediction:", prediction)

# print(classes[np.argmax(prediction)])

# # confidence
# print(np.max(prediction) * 100)
