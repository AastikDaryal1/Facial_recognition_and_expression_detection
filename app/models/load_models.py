import pickle
import tensorflow as tf
import json

def load_face_model():
    with open("saved_models/team_face_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("saved_models/team_label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


def load_emotion_model():
    model = tf.keras.models.load_model("saved_models/cnn_expression_model.h5")

    with open("saved_models/class_names.json") as f:
        classes = json.load(f)

    return model, classes