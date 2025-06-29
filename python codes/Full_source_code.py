# -*- coding: utf-8 -*-
"""ASL detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HpTwbxH5ps7FtJN-XeOF9qcfsFGuH0jP
"""

!pip install mediapipe opencv-python pandas tqdm

import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!pip install -q kaggle

!kaggle datasets download -d grassknoted/asl-alphabet

!unzip -q asl-alphabet.zip -d asl_data

import os
import cv2
import pickle
import mediapipe as mp

# Path to the ASL dataset folder
DATA_DIR = '/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train'

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Output storage
data = []
labels = []

# Limit to speed up
max_images_per_class = 300

# Process each class folder
for dir_ in sorted(os.listdir(DATA_DIR)):
    print(f"\n🔤 Processing letter: {dir_}")
    img_count = 0

    for img_name in os.listdir(os.path.join(DATA_DIR, dir_)):
        if img_count >= max_images_per_class:
            break

        img_path = os.path.join(DATA_DIR, dir_, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip broken images

        # Resize and convert to RGB
        img = cv2.resize(img, (256, 256))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Process only the first detected hand to maintain consistent feature vector length
            hand_landmarks = results.multi_hand_landmarks[0]
            data_aux = []
            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
            img_count += 1

        if img_count % 50 == 0 and img_count > 0:
            print(f"  ✅ {img_count} images processed for '{dir_}'")

print(f"\n✅ Total samples collected: {len(data)}")

# Save as pickle file
with open('asl_landmarks.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\n📦 Saved landmark dataset as 'asl_landmarks.pickle'")

import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load your landmark data
with open('asl_landmarks.pickle', 'rb') as f:
    dataset = pickle.load(f)

X = dataset['data']
y = dataset['labels']

# Check if all data points have the same length
lengths = [len(x) for x in X]
if len(set(lengths)) > 1:
    print(f"Error: Data points have inconsistent lengths. Found lengths: {set(lengths)}")
    print("This is likely due to variations in the number of hands detected in the images.")
    print("Please revisit the data collection step (cell SYneibg6OZmK) to ensure consistent data representation.")
else:
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    with open('asl_knn_model.pickle', 'wb') as f:
        pickle.dump(model, f)

    print("\n💾 Model saved as 'asl_knn_model.pickle'")

import cv2
import pickle
import mediapipe as mp

# Load trained model
with open('asl_knn_model.pickle', 'rb') as f:
    model = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Predict
            prediction = model.predict([data_aux])[0]

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show prediction
            cv2.putText(frame, prediction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()