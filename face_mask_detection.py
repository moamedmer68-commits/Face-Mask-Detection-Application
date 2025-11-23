import streamlit as st
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.title("Face Mask Detection — HuggingFace Model")

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("gilbertoferreira/face-mask-detection")
    model = AutoModelForImageClassification.from_pretrained("gilbertoferreira/face-mask-detection")
    return processor, model

processor, model = load_model()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def detect_and_predict_mask(image):
    # Haar cascade face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]

        # HF model input
        inputs = processor(images=face, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_id = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[predicted_id]

        results.append({"box": (x, y, w, h), "label": label})

    return results

if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    detections = detect_and_predict_mask(image)

    for det in detections:
        x, y, w, h = det["box"]
        label = det["label"]
        color = (0, 255, 0) if "mask" in label.lower() else (255, 0, 0)

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            image, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    st.image(image, channels="BGR", use_column_width=True)
