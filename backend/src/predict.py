import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/brain_tumor_model.h5")

# Force model build
dummy = np.zeros((1, 224, 224, 3))
model.predict(dummy)

class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(img_path):
    img_array = preprocess_image(img_path)

    predictions = model.predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence = float(np.max(predictions) * 100)

    # Multi-class probabilities
    probabilities = {
        class_labels[i]: float(predictions[i] * 100)
        for i in range(len(class_labels))
    }

    # -----------------------
    # Binary Classification
    # -----------------------

    if predicted_class == "No Tumor":
        binary_prediction = 0
        binary_label = "No Tumor"
    else:
        binary_prediction = 1
        binary_label = "Tumor Detected"

    return predicted_class, confidence, probabilities, None, binary_prediction, binary_label