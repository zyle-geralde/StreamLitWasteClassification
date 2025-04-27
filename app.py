import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

#load model
model = load_model('asl_resnet50_model.h5')

#set image size
IMG_HEIGHT, IMG_WIDTH = 128, 128

#class labels
class_labels = ['O', 'R']

#streamlit page configuration
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")
st.title("♻️ Waste Classification (Organic vs Recyclable)")

#mode selection
mode = st.radio("Choose input method:", ('Upload an Image', 'Take a Photo'))

def predict_image(img):
    img = img.convert('RGB')# ensure RGB
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)
    
    return predicted_class, confidence

if mode == 'Upload an Image':
    uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        predicted_class, confidence = predict_image(img)

        #display results
        st.subheader("Prediction Result")
        readable_class = "Organic" if predicted_class == "O" else "Recyclable"
        st.write(f"**Prediction:** {readable_class} ({confidence}%)")

elif mode == 'Take a Photo':
    picture = st.camera_input("Take a picture")

    if picture is not None:
        img = Image.open(picture)
        st.image(img, caption='Captured Image', use_column_width=True)

        predicted_class, confidence = predict_image(img)

        #display results
        st.subheader("Prediction Result")
        readable_class = "Organic" if predicted_class == "O" else "Recyclable"
        st.write(f"**Prediction:** {readable_class} ({confidence}%)")
