#  Face Mask Detection Application  

##  Overview  
This project is a **Face Mask Detection App** built with **Streamlit**, **OpenCV**, and **TensorFlow/Keras**.  
The app uses a pre-trained deep learning model to detect whether people in an uploaded image are **wearing a mask** or **not wearing a mask**.  

---

##  Features  
   Upload an image and automatically detect faces   
   Predict whether each detected face is wearing a mask or not  
   Highlight detected faces with colored bounding boxes:  
-  **Mask** → Green box  
-  **No Mask** → Red box  
  Simple, interactive Streamlit UI  

---

##  Technologies Used  
- **Python 3.8+**  
- **Streamlit** – web interface  
- **OpenCV** – face detection (Haar Cascade)  
- **TensorFlow / Keras** – deep learning model  
- **NumPy** – image preprocessing  
- **Pillow (PIL)** – image handling  

---

## Installation  

###  Clone the repository  
```bash
git clone https://github.com/your-username/face-mask-detection-app.git
cd face-mask-detection-app

2️ Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate       # For Linux/Mac
venv\Scripts\activate          # For Windows
3️ Install dependencies
streamlit
opencv-python
tensorflow
numpy
Pillow
