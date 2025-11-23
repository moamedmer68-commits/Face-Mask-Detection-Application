import streamlit as st
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForImageClassification
import torch

st.title("Face Mask Detection — HuggingFace Model")

# Load model from Hugging Face (cached to avoid downloading every run)
@st.cache_resource
def load_hf_model():
    # Load processor and model from HF Hub
    processor = AutoProcessor.from_pretrained("prithivMLmods/Face-Mask-Detection")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Face-Mask-Detection")
    return processor, model

processor, model = load_hf_model()
st.success("Model loaded successfully from Hugging Face")

# File uploader for image input
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def detect_and_predict_mask(image):
    """
    Detect faces using OpenCV Haar Cascade,
    then run each detected face through the HuggingFace model.
    """
    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []
    for (x, y, w, h) in faces:
        # Crop the face region
        face_img = image[y:y+h, x:x+w]

        # Prepare input for the Hugging Face model
        inputs = processor(images=face_img, return_tensors="pt")

        # Run the model (no gradients needed)
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get the predicted class ID
        predicted_id = torch.argmax(logits, dim=-1).item()

        # Convert ID to readable label
        label = model.config.id2label[predicted_id]

        results.append({"box": (x, y, w, h), "label": label})

    return results


if uploaded_img:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Detect faces + predict mask or no mask
    detections = detect_and_predict_mask(image)

    # Draw bounding boxes and labels
    for det in detections:
        x, y, w, h = det["box"]
        label = det["label"]

        # Green if mask detected, red if no mask
        color = (0, 255, 0) if "Mask" in label else (0, 0, 255)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display final image
    st.image(image, channels="BGR", use_column_width=True)



