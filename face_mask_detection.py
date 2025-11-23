import os

# Install required packages if not already installed
os.system("pip install transformers")
os.system("pip install torch")
os.system("pip install streamlit")
os.system("pip install opencv-python-headless")  # headless version works better in server env
os.system("pip install numpy")

# -----------------------------------------------------------
# Import packages after installation
# -----------------------------------------------------------
import streamlit as st
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForImageClassification
import torch

# -----------------------------------------------------------
# Streamlit App Title
# -----------------------------------------------------------
st.title("Face Mask Detection — HuggingFace Model")

# -----------------------------------------------------------
# Load HuggingFace Model (cached)
# -----------------------------------------------------------
@st.cache_resource
def load_hf_model():
    processor = AutoProcessor.from_pretrained("prithivMLmods/Face-Mask-Detection")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Face-Mask-Detection")
    return processor, model

processor, model = load_hf_model()
st.success("Model loaded successfully from Hugging Face")

# -----------------------------------------------------------
# File Upload
# -----------------------------------------------------------
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------------------------------------
# Face Detection + Mask Prediction
# -----------------------------------------------------------
def detect_and_predict_mask(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        inputs = processor(images=face_img, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_id = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[predicted_id]
        results.append({"box": (x, y, w, h), "label": label})

    return results

# -----------------------------------------------------------
# Process uploaded image
# -----------------------------------------------------------
if uploaded_img:
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    results = detect_and_predict_mask(image)

    for det in results:
        x, y, w, h = det["box"]
        label = det["label"]
        color = (0, 255, 0) if "Mask" in label else (255, 0, 0)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    st.image(image, channels="BGR", use_column_width=True)

