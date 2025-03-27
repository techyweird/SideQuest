import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load trained model
with open('emotion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Load label encoder
label_encoder = LabelEncoder()
try:
    label_encoder.classes_ = np.load("label_classes.npy", allow_pickle=True)
except FileNotFoundError:
    print("❌ Missing label_classes.npy! Run the training script again to generate it.")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

def normalize_confidence(confidence):
    return round((1 / (1 + np.exp(-confidence))) * 100, 2)  # Sigmoid normalization

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                x, y = lm.x, lm.y  # Normalized coordinates
                landmarks.extend([x, y])

            landmarks = np.array(landmarks).reshape(1, -1)  # Reshape for model
            probabilities = model.decision_function(landmarks)  # Get confidence scores
            predicted_label = model.predict(landmarks)[0]  # Predict emotion

            # Convert scores to percentages with sigmoid normalization
            confidence = max(probabilities)  # Get highest confidence
            confidence_percent = normalize_confidence(confidence)

            # Display emotion & confidence on screen
            text = f"Emotion: {predicted_label} ({confidence_percent}%)"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the video feed
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
