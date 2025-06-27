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