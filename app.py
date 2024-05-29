import streamlit as st
import numpy as np
import pandas as pd


# Streamlit app
st.title("Emotion Detector")

# Upload file section
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = f'temp/{uploaded_file.name}'
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    model = load_model()
    display_image_with_prediction(model, img_path)
else:
    st.write("Please upload an image file.")
