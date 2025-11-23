import streamlit as st
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForImageClassification
import torch

# -----------------------------------------------------------
# Streamlit App Title
# -----------------------------------------------------------
st.title("Face Mask Detection — HuggingFace Transformers")

# -----------------------------------------------------------
# Load HuggingFace Model (with caching)
# -----------------------------------------------------------
@st.cache_resource
def load_hf_model():
    """
    Load the processor and model from Hugging Face Hub.
    Cached to avoid re-downloading.
    """
    processor = AutoProcessor.from_pretrained("prithivMLmods/Face-Mask-Detection")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Face-Mask-Detection")
    return processor, model

processor, model = load_hf_model()
st.success("Model loaded successfully from Hugging Face")

# -----------------------------------------------------------
# File Upload Area
# -----------------------------------------------------------
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------------------------------------
# Face Detection and Prediction
# -----------------------------------------------------------
def detect_and_predict_mask(image):
    """
    Detect faces using Haar Cascade,
    then classify each face using HuggingFace model.
    """
    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    detections = []
    for (x, y, w, h) in faces:
        # Crop the face
        face_img = image[y:y+h, x:x+w]

        # Prepare for model
        inputs = processor(images=face_img, return_tensors="pt")

        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits

        class_id = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[class_id]

        detections.append({"box": (x, y, w, h), "label": label})

    return detections

# -----------------------------------------------------------
# Process uploaded image
# -----------------------------------------------------------
if uploaded_img:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Detect + classify
    results = detect_and_predict_mask(image)

    # Draw results
    for det in results:
        x, y, w, h = det["box"]
        label = det["label"]

        # Green = Mask, Red = No Mask
        color = (0, 255, 0) if "Mask" in label else (255, 0, 0)

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            image, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    st.image(image, channels="BGR", use_column_width=True)
