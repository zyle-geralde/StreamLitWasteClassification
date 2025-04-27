
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

# Load model
model = load_model('asl_resnet50_model.h5')  # Your trained model

# Image size
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Class labels
class_labels = ['O', 'R']

# Streamlit page config
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")
st.title("♻️ Waste Classification (Organic vs Recyclable)")

# Mode selection
mode = st.radio("Choose input method:", ('Upload an Image', 'Use Webcam'))

if mode == 'Upload an Image':
    uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        #temporarily save file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        #preprocess image
        img = Image.open(temp_file_path)
        img = img.convert('RGB')  # Just in case
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        #prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = round(np.max(predictions) * 100, 2)

        #display result
        st.subheader("Prediction Result")
        readable_class = "Organic" if predicted_class == "O" else "Recyclable"
        st.write(f"**Prediction:** {readable_class} ({confidence}%)")

elif mode == 'Use Webcam':
    run = st.checkbox('Start Webcam')

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error('Failed to grab frame.')
            break

        #process camera
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        #prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = round(np.max(predictions) * 100, 2)

        #Overlay prediction frame
        label = f"{'Organic' if predicted_class == 'O' else 'Recyclable'} ({confidence}%)"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        #frame update
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()

