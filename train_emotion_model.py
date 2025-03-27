import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Define dataset path
data_path = "dataset/"  # Ensure this folder exists
emotions = os.listdir(data_path)  # Get emotion labels

X = []  # Features (landmarks)
y_labels = []  # Emotion labels

# Process images
for emotion in emotions:
    folder = os.path.join(data_path, emotion)

    if not os.path.isdir(folder):  # Ensure it's a directory
        continue

    print(f"📂 Processing category: {emotion}")

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ Skipping {img_path} (image not found)")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    x, y = lm.x, lm.y  # Normalized coordinates
                    landmarks.extend([x, y])

                X.append(landmarks)
                y_labels.append(str(emotion))  # Ensure it's a string

# Convert to numpy arrays
X = np.array(X, dtype=np.float32)
y_labels = np.array(y_labels, dtype=str)

# ✅ Debugging: Check dataset size
print(f"\n✅ Total Samples: {len(X)}")
if len(X) == 0:
    print("❌ No face landmarks detected! Check dataset images.")
    exit()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

# Train the SVM Model
print("🛠️ Training model...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)


# After training the model
label_encoder = LabelEncoder()
label_encoder.fit(y_labels)  # Fit encoder with emotion labels

# Save label classes
np.save("label_classes.npy", label_encoder.classes_)


# Evaluate Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save Model
try:
    with open('emotion_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model successfully saved as emotion_model.pkl")
except Exception as e:
    print(f"❌ Failed to save model: {e}")
