import cv2
import numpy as np

def predict_emotion(image_path, model, classes):
    img = cv2.imread(image_path)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = preds.argmax()

    return {
        "emotion": classes[idx],
        "confidence": float(preds[idx])
    }