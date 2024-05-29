import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Fungsi untuk memuat dan menampilkan gambar serta melakukan prediksi
def load_model():
    model = tf.keras.models.load_model('emotion_model.h5')
    return model

def predict_emotion(model, img_path):
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    emotion_labels = ['Angry', 'Happy', 'Sad']
    predicted_class = np.argmax(prediction)
    return emotion_labels[predicted_class]

def display_image_with_prediction(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224), interpolation='lanczos')
    img_array = image.img_to_array(img)
    img_array = img_array * 0.8
    img_array = np.clip(img_array, 0, 255)
    
    plt.imshow(img_array.astype(np.uint8))
    plt.axis('off')
    st.pyplot(plt)

    predicted_emotion = predict_emotion(model, img_path)
    st.write(f'Hasil Prediksi: {predicted_emotion}')

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
