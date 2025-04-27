import streamlit as st
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model('asl_resnet50_model.h5')  # Rename accordingly

# Define image size (must match your training image size)
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Class labels (ensure the order matches your training generator class_indices)
class_labels = ['O', 'R']

# Streamlit app
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")

st.title("♻️ Waste Classification (Organic vs Recyclable)")

uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Preprocess the image
    img = image.load_img(temp_file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale like during training
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)

    # Display results
    st.subheader("Prediction Result")
    readable_class = "Organic" if predicted_class == "O" else "Recyclable"
    st.write(f"**Prediction:** {readable_class}")
