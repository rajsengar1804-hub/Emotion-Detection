import json
import os
import numpy as np
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "emotion_final_model.keras"
LABEL_MAP_PATH = "label_mapping.json"
IMG_SIZE = 224
test_image_path = "test_face.png"

# -------------------------------
# Download model from Hugging Face
# -------------------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")

    hf_hub_download(
        repo_id="rajsengar9340/emotion_detection",
        filename="emotion_final_model.keras",
        local_dir="."
    )

    print("Model downloaded successfully!")

# -------------------------------
# Load model
# -------------------------------
model = load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_mapping = json.load(f)
    label_mapping = {int(k): v for k, v in label_mapping.items()}


def preprocess_face_array(face_img_gray):
    h, w = face_img_gray.shape
    size = max(h, w)
    # Pad to square using edge pixels, centered
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    square_img = cv2.copyMakeBorder(
        face_img_gray, pad_h, size - h - pad_h, pad_w, size - w - pad_w,
        borderType=cv2.BORDER_REPLICATE
    )
    img_small = cv2.resize(square_img, (48, 48))
    img = cv2.resize(img_small, (IMG_SIZE, IMG_SIZE))
    img_array = img.astype("float32")
    img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_emotion_from_array(face_img_gray):
    processed = preprocess_face_array(face_img_gray)
    predictions = model(processed, training=False).numpy()[0]
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])
    emotion = label_mapping[predicted_idx]
    return emotion, confidence


def predict_all_probabilities(face_img_gray):
    processed = preprocess_face_array(face_img_gray)
    predictions = model(processed, training=False).numpy()[0]
    probs = {label_mapping[i]: float(predictions[i]) for i in range(len(predictions))}
    return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype("float32")
    img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_emotion(image_path):
    processed = preprocess_image(image_path)
    predictions = model(processed, training=False).numpy()[0]
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])
    emotion = label_mapping[predicted_idx]
    return emotion, confidence


if __name__ == "__main__":
    emotion, confidence = predict_emotion(test_image_path)
    print(f"Predicted emotion: {emotion} ({confidence:.2%} confidence)")tus