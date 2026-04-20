import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

TRAIN_PATH = "data/raf-db/train"
TEST_PATH = "data/raf-db/test"

IMG_SIZE = 224
BATCH_SIZE = 16

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=5)

os.makedirs("saved_models", exist_ok=True)

model.save("saved_models/cnn_expression_model.h5")

with open("saved_models/class_names.json", "w") as f:
    json.dump(list(train_gen.class_indices.keys()), f)

print("✅ Emotion model trained")