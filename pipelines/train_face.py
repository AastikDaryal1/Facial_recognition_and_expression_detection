import os
import pickle
import numpy as np
import cv2
from deepface import DeepFace
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

TEAM_DIR = "data/team_faces"
SAVE_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)

embeddings = []
labels = []

for person in os.listdir(TEAM_DIR):
    person_path = os.path.join(TEAM_DIR, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            emb = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append(emb)
            labels.append(person)

        except:
            continue

X = normalize(np.array(embeddings))
y = np.array(labels)

le = LabelEncoder()
y_enc = le.fit_transform(y)

model = CalibratedClassifierCV(SVC(kernel="rbf", C=10))
model.fit(X, y_enc)

with open("saved_models/team_face_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("saved_models/team_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Face model trained")