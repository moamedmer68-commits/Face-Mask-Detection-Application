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
# Load HuggingFace Model (cached so it loads only once)
# -----------------------------------------------------------
@st.cache_resource
def load_hf_model():
    """
    Load the processor and model from Hugging Face Hub.
    Cached to improve performance in Streamlit.
    """
    processor = AutoProcessor.from_pretrained("prithivMLmods/Face-Mask-Detection")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Face-Mask-Detection")
    return processor, model

processor, model = load_hf_model()
st.success("Model loaded successfully from Hugging Face")

# -----------------------------------------------------------
# File Upload Section
# -----------------------------------------------------------
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------------------------------------
# Face Detection + Mask Prediction
# -----------------------------------------------------------
def detect_and_predict_mask(image):
    """
    Detect faces using OpenCV Haar Cascade,
    then classify each face using the HuggingFace model.
    """
    # Initialize Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    detections = []
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = image[y:y+h, x:x+w]

        # Prepare input for HuggingFace model
        inputs = processor(images=face_img, return_tensors="pt")

        # Predict using the model
        with torch.no_grad():
            logits = model(**inputs).logits

        # Convert prediction to label
        predicted_class = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[predicted_class]

        detections.append({"box": (x, y, w, h), "label": label})

    return detections

# -----------------------------------------------------------
# Process image if user uploads one
# -----------------------------------------------------------
if uploaded_img:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Run detection + prediction
    results = detect_and_predict_mask(image)

    # Draw bounding boxes and labels
    for det in results:
        x, y, w, h = det["box"]
        label = det["label"]

        # Green for mask, red for no mask
        color = (0, 255, 0) if "Mask" in label else (255, 0, 0)

        # Draw rectangle and label
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    # Display final image
    st.image(image, channels="BGR", use_column_width=True)

