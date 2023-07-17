"""
 作者 lgf
 日期 2023/3/27
"""
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Load model
model = load_model(r'C:\Users\小凡\Downloads\converted_keras\alphabet.h5',compile=False)
# Load the labels
class_names = open(r'C:\Users\小凡\Downloads\converted_keras\alphabet.txt',"r").readlines()
# Load Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Load and preprocess image
image_path = 'image.jpg'
image = cv2.imread(image_path)
if image is None:
    print('Failed to read image.')
else:
    # Detect hand landmarks
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        # Extract hand bounding box
        hand_landmarks = results.multi_hand_landmarks[0]
        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        for landmark in hand_landmarks.landmark:
            x, y = landmark.x, landmark.y
            x_min, x_max = min(x_min, x), max(x_max, x)
            y_min, y_max = min(y_min, y), max(y_max, y)
        x_min, x_max, y_min, y_max = int(x_min * image.shape[1]), int(x_max * image.shape[1]), int(y_min * image.shape[0]), int(y_max * image.shape[0])
        # Crop image to hand bounding box
        hand_image = image[y_min:y_max, x_min:x_max]
        # Resize image to model input shape
        hand_image = cv2.resize(hand_image, (224, 224), interpolation=cv2.INTER_AREA)
        # Convert image to float32 and normalize
        hand_image = np.asarray(hand_image, dtype=np.float32)
        hand_image = (hand_image / 255.0) - 0.5
        hand_image = np.expand_dims(hand_image, axis=0)
        # Predict class probabilities
        class_probabilities = model.predict(hand_image)[0]
        # Get predicted class index and name
        predicted_class_index = np.argmax(class_probabilities)
        predicted_class_name = class_names[predicted_class_index].strip()
        # Print predicted class name and probabilities
        print('Predicted class: {}'.format(predicted_class_name))
        for i in range(len(class_names)):
            class_name = class_names[i].strip()
            probability = class_probabilities[i]
            print('{} probability: {:.2f}%'.format(class_name, probability * 100))
    else:
        print('No hand detected in image.')