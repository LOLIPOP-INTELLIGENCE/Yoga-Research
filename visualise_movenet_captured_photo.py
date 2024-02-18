import pandas as pd
import cv2
from matplotlib import pyplot as plt

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
image_path = 'captured_photo.png'
print(image_path)

image = cv2.imread(image_path)

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
