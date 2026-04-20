from deepface import DeepFace
from sklearn.preprocessing import normalize

def recognize_face(image_path, model, encoder):
    emb = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]

    emb = normalize([emb])
    probs = model.predict_proba(emb)[0]
    idx = probs.argmax()

    return {
        "name": encoder.classes_[idx],
        "confidence": float(probs[idx])
    }