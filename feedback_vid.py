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
import time
import math
import json 
import sys

from openai import OpenAI

api_key = "test_key"
client = OpenAI(api_key=api_key)

model = load_model('yoga_poses_5/weights.best.hdf5')

cap = cv2.VideoCapture(0)
classes = ['chair', 'tree', 'dog', 'cobra', 'warrior']

correct_angles = []
# append chair poses
correct_angles.append({"right_hip_right_knee_right_ankle": 114.8326986831958, "left_hip_left_knee_left_ankle": 116.59334914919609, "right_knee_right_hip_right_shoulder": 100.34898087445153, "left_knee_left_hip_left_shoulder": 101.91423721006468, "right_hip_right_shoulder_right_elbow": 163.43969438516714, "left_hip_left_shoulder_left_elbow": 165.43235435034882, "right_shoulder_right_elbow_right_wrist": 167.3188552744455, "left_shoulder_left_elbow_left_wrist": 168.40712770488568})
# append tree poses
correct_angles.append({"right_hip_right_knee_right_ankle": 173.74329151734284, "left_hip_left_knee_left_ankle": 29.694201130350134, "right_knee_right_hip_right_shoulder": 179.8721337942167, "left_knee_left_hip_left_shoulder": 116.16944318761996, "right_hip_right_shoulder_right_elbow": 24.569462400306463, "left_hip_left_shoulder_left_elbow": 21.24259620450775, "right_shoulder_right_elbow_right_wrist": 48.35206594481974, "left_shoulder_left_elbow_left_wrist": 38.254843682288104})
# append dog poses
correct_angles.append({"right_hip_right_knee_right_ankle": 175.47644365424046, "left_hip_left_knee_left_ankle": 178.0277358881745, "right_knee_right_hip_right_shoulder": 75.3786617433573, "left_knee_left_hip_left_shoulder": 75.52774060045532, "right_hip_right_shoulder_right_elbow": 154.78095554700687, "left_hip_left_shoulder_left_elbow": 158.19293613889965, "right_shoulder_right_elbow_right_wrist": 145.56627563761054, "left_shoulder_left_elbow_left_wrist": 151.22423104216344})
# append cobra poses
correct_angles.append({"right_hip_right_knee_right_ankle": 167.28636783672698, "left_hip_left_knee_left_ankle": 168.146995832256, "right_knee_right_hip_right_shoulder": 121.148580986195, "left_knee_left_hip_left_shoulder": 119.02890778349075, "right_hip_right_shoulder_right_elbow": 19.431296719426953, "left_hip_left_shoulder_left_elbow": 18.657360786633816, "right_shoulder_right_elbow_right_wrist": 155.912753890314, "left_shoulder_left_elbow_left_wrist": 159.06530969154343})
# append warrior poses
correct_angles.append({"right_hip_right_knee_right_ankle": 157.43061506273932, "left_hip_left_knee_left_ankle": 166.9180874873042, "right_knee_right_hip_right_shoulder": 80.25898616240707, "left_knee_left_hip_left_shoulder": 174.74264496133566, "right_hip_right_shoulder_right_elbow": 160.22148568709903, "left_hip_left_shoulder_left_elbow": 164.39792513346904, "right_shoulder_right_elbow_right_wrist": 176.7006310213317, "left_shoulder_left_elbow_left_wrist": 174.94013115373588})

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
    person = movenet.detect(input_tensor, reset_crop_region=False)

  return person

def calculate_angle(A, B, C):
    # Adjust points for the inverted y-axis: invert the y-component
    BA = (A[0] - B[0], B[1] - A[1])  # Invert y-component
    BC = (C[0] - B[0], B[1] - C[1])  # Invert y-component
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    magnitude_AB = math.sqrt(BA[0]**2 + BA[1]**2)
    magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)
    cos_theta = dot_product / (magnitude_AB * magnitude_BC)
    cos_theta = max(min(cos_theta, 1), -1)
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def create_angles(row):
    right_shoulder = 6
    right_elbow = 8
    right_wrist = 10
    right_hip = 12
    right_knee = 14
    right_ankle = 16

    right_shoulder_point = (int(row[right_shoulder * 3]), int(row[right_shoulder * 3 + 1]))
    right_elbow_point = (int(row[right_elbow * 3]), int(row[right_elbow * 3 + 1]))
    right_wrist_point = (int(row[right_wrist * 3]), int(row[right_wrist * 3 + 1]))
    right_hip_point = (int(row[right_hip * 3]), int(row[right_hip * 3 + 1]))
    right_knee_point = (int(row[right_knee * 3]), int(row[right_knee * 3 + 1]))
    right_ankle_point = (int(row[right_ankle * 3]), int(row[right_ankle * 3 + 1]))

    left_shoulder = 5
    left_elbow = 7
    left_wrist = 9
    left_hip = 11
    left_knee = 13
    left_ankle = 15

    left_shoulder_point = (int(row[left_shoulder * 3]), int(row[left_shoulder * 3 + 1]))
    left_elbow_point = (int(row[left_elbow * 3]), int(row[left_elbow * 3 + 1]))
    left_wrist_point = (int(row[left_wrist * 3]), int(row[left_wrist * 3 + 1]))
    left_hip_point = (int(row[left_hip * 3]), int(row[left_hip * 3 + 1]))
    left_knee_point = (int(row[left_knee * 3]), int(row[left_knee * 3 + 1]))
    left_ankle_point = (int(row[left_ankle * 3]), int(row[left_ankle * 3 + 1]))

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

    return all_angles

def compare_angles(correct_angles, angles):
    sum = 0
    for key in correct_angles:
        sum += abs(correct_angles[key] - angles[key])
    return sum

def give_feedback (correct_angles, min_angles, pose_name):

    text_str = "\n\nThe format of the following is correct angle -> my angle\n"
    for key in correct_angles:
        text_str += '\n' + (f"{key}: {correct_angles[key]} -> {min_angles[key]}")

    print(pose_name)
    print(text_str)

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a Yoga Teacher."},
            {"role": "user", "content": f"I am trying to perform the {pose_name} pose. I have a few angles that I recorded professionals doing. I also recorded my angles while doing this pose. But I donn't understand how to compare the angles. I don't want to know how many degrees I was off by but rather what I could have done in terms of bending or stretching to get closer to the correct angle. Can you help me with that? Here are the angles I recorded: {text_str} Your response should be short and crisp."},
        ]
    )


    print(response.choices[0].message.content)
    # exit
    sys.exit()

    # print("")

    # for key in correct_angles:
    #     if correct_angles[key] - 5 <= min_angles[key] <= correct_angles[key] + 5:
    #         print(f"{key} is correct")
    #     elif correct_angles[key] - 10 <= min_angles[key] <= correct_angles[key] + 10:
    #         print(f"{key} is almost correct")
    #     elif correct_angles[key] - 15 <= min_angles[key] <= correct_angles[key] + 15:
    #         print(f"{key} is incorrect")
    #     else:
    #         print(f"{key} is very incorrect")






previous_pose = None
previous_pose_start_time = None
current_pose = None
current_pose_start_time = None

min_sum = 1000000000
min_pose = None
min_angles = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    person = detect(frame)

    pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score] for keypoint in person.keypoints], dtype=np.float32)
    coordinates = pose_landmarks.flatten()

    if len(coordinates) == 51:
        x = coordinates.reshape(1, 51)
        prediction = model.predict(x)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # If the predicted pose is the same as the current pose being tracked
        if predicted_class == current_pose:
            # Check if this pose has been held for at least 2 seconds
            if time.time() - current_pose_start_time >= 2:
                # Only announce the pose change once after it's been held for 2 seconds
                if current_pose != previous_pose:
                    if min_pose is not None:
                        give_feedback(curr_correct_angle, min_angles, previous_pose)                    
                    if previous_pose is not None:
                        print(f"{previous_pose} pose ended.")
                        min_sum = 1000000000
                        min_pose = None
                        min_angles = None                        
                    print(f"{predicted_class} pose started and held for 2 seconds.")
                    previous_pose = predicted_class
                    previous_pose_start_time = current_pose_start_time

                    curr_correct_angle = correct_angles[np.argmax(prediction)]

                
                # convert each element of coordinates to float
                for i in range(len(coordinates)):
                    coordinates[i] = float(coordinates[i])

                # create angles
                angles = create_angles(coordinates)

                # compare angles
                curr_sum = compare_angles(curr_correct_angle, angles)
                if curr_sum < min_sum:
                    min_sum = curr_sum
                    # save current frame in min_pose
                    min_pose = frame
                    # save the image
                    cv2.imwrite('min_pose.jpg', min_pose)
                    min_angles = angles
                    # save angles in a json
                    with open('min_angles.json', 'w') as f:
                        json.dump(min_angles, f)
                print(curr_sum)
                print(min_angles == None)


                


        else:
            # If a new pose is detected, start tracking it
            current_pose = predicted_class
            current_pose_start_time = time.time()

        cv2.putText(frame, f'{predicted_class} ({confidence:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        print('Incorrect number of keypoints detected')
    
    cv2.imshow('Pose Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()