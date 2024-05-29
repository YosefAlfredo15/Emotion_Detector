import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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
