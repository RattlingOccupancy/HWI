
import os
# Suppress TensorFlow oneDNN warnings
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np # type: ignore

# Import TensorFlow and Keras for model creation and conversion
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore

# ---------------------------
# Step 1: Build and Save a Simple Model
# ---------------------------
# Create a simple model with an explicit Input layer
model = keras.Sequential([
    keras.Input(shape=(10,)),  # Explicit input layer
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save model using the recommended native Keras format
model.save("your_model.keras")
print("Model saved as your_model.keras")

# ---------------------------
# Step 2: Convert the Keras Model to TFLite
# ---------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True  # Force use of the new converter
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # Use built-in ops only
tflite_model = converter.convert()

# Save the converted TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model converted and saved as model.tflite")

# ---------------------------
# Step 3: Initialize MediaPipe Face Mesh for Face & Eye Tracking
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------
# Step 4: Define Function to Detect Face, Eyes, and Analyze Movement
# ---------------------------
def detect_cheating(frame):
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get key facial landmarks for head and eye tracking
            nose_tip = face_landmarks.landmark[1]  # Nose tip
            left_eye = face_landmarks.landmark[33]   # Left eye
            right_eye = face_landmarks.landmark[263] # Right eye

            # Convert normalized coordinates to pixel coordinates
            nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
            left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
            right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

            # Draw key points on the frame
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (left_eye_x, left_eye_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (right_eye_x, right_eye_y), 5, (255, 0, 0), -1)

            # Determine head movement direction using nose tip position relative to frame center
            center_x = w // 2
            if nose_x < center_x - 50:
                head_position = "Looking Left"
            elif nose_x > center_x + 50:
                head_position = "Looking Right"
            else:
                head_position = "Centered"

            # Determine eye movement by comparing the midpoint of both eyes to the center of the frame
            eye_mid_x = (left_eye_x + right_eye_x) // 2
            if eye_mid_x < center_x - 50:
                eye_position = "Eyes Left"
            elif eye_mid_x > center_x + 50:
                eye_position = "Eyes Right"
            else:
                eye_position = "Eyes Forward"

            # Flag potential cheating if either head or eyes deviate from the center
            if head_position != "Centered" or eye_position != "Eyes Forward":
                status = "Potential Cheating 🚨"
                color = (0, 0, 255)  # Red
            else:
                status = "Focused ✅"
                color = (0, 255, 0)  # Green

            # Overlay the status on the frame
            cv2.putText(frame, f"Head: {head_position}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Eyes: {eye_position}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    return frame

# ---------------------------
# Step 5: Run Real-Time Video Capture & Detection
# ---------------------------
cap = cv2.VideoCapture(0)  # Open webcam (ensure it’s accessible)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame: detect face, eyes, and check for cheating behavior
    processed_frame = detect_cheating(frame)

    # Display the output window
    cv2.imshow("AI Interview Cheating Detection", processed_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
