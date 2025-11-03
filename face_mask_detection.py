import os
import gdown
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title('Face Mask Detection Application')
os.system("pip install opencv-python")

# Model local filename
model_path = "face_mask_detection_model.h5"

# If model file does not exist locally -> download it from Google Drive
if not os.path.exists(model_path):
    # IMPORTANT: replace this with the actual FILE ID of the MODEL (.h5) not the folder ID
    file_id = "1PRgcbaVB7jHSD6_HhQHJW9fwZ0VgOWIS"
    url = f"https://drive.google.com/uc?id={file_id}"

    st.write("Downloading model file from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)
st.success("Model loaded successfully")

# File Uploader
upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def detect_and_predict_mask(image):
    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1)

    predictions = []

    for (x, y, w, h) in faces:
        # Crop face region
        face = image[y:y+h, x:x+w]

        # Convert BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        face = cv2.resize(face, (128, 128))

        # Normalize
        face = np.array(face) / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Predict
        predictions.append(model.predict(face))
    
    return faces, predictions

if upload is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces, preds = detect_and_predict_mask(image)

    # Draw result on image
    for i, (x, y, w, h) in enumerate(faces):
        (mask, withoutMask) = preds[i][0]
        label = "Mask" if mask > withoutMask else "No Mask"

        # Green for mask, red for no mask
        color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    st.image(image, use_column_width=True)
