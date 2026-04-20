from fastapi import FastAPI, UploadFile, File
import shutil

from app.models.load_models import load_face_model, load_emotion_model
from app.services.face_service import recognize_face
from app.services.emotion_service import predict_emotion

app = FastAPI()

face_model, face_encoder = load_face_model()
emotion_model, emotion_classes = load_emotion_model()

@app.get("/")
def home():
    return {"message": "Face + Emotion API running 🚀"}

@app.post("/predict/face")
async def predict_face(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return recognize_face(path, face_model, face_encoder)

@app.post("/predict/emotion")
async def predict_emotion_api(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return predict_emotion(path, emotion_model, emotion_classes)