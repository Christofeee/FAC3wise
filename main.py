import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load pretrained models for age and gender detection
age_model_path = 'models/age_net.caffemodel'
age_prototxt_path = 'prototxts/age_deploy.prototxt'
gender_model_path = 'models/gender_net.caffemodel'
gender_prototxt_path = 'prototxts/gender_deploy.prototxt'

age_net = cv2.dnn.readNet(age_model_path, age_prototxt_path)
gender_net = cv2.dnn.readNet(gender_model_path, gender_prototxt_path)

# Define age and gender labels
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# Create Streamlit UI
st.title("Age and Gender Detection")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Perform age and gender detection on the uploaded image
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Age detection
    age_net.setInput(blob)
    age_predictions = age_net.forward()
    age = age_labels[np.argmax(age_predictions)]

    # Gender detection
    gender_net.setInput(blob)
    gender_predictions = gender_net.forward()
    gender = gender_labels[np.argmax(gender_predictions)]

    # Display the image with predictions
    st.image(image, caption="Predicted Age and Gender", use_column_width=True)

    # Display age and gender predictions
    st.write(f"Predicted Age: {age}")
    st.write(f"Predicted Gender: {gender}")
