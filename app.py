import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tempfile
import matplotlib.pyplot as plt

# Fungsi untuk memuat dan menampilkan gambar serta melakukan prediksi
def load_model():
    model = tf.keras.models.load_model('emotion_model.h5')
    return model

def predict_emotion(model, img):
    img_array = np.array(img.resize((48, 48)).convert('L'))  # Convert ke skala abu-abu dan resize gambar
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # Tambahkan dimensi kedua dan keempat
    img_array = img_array / 255.0  # Normalisasi
    prediction = model.predict(img_array)
    emotion_labels = ['Angry', 'Happy', 'Sad']
    predicted_class = np.argmax(prediction)
    return emotion_labels[predicted_class]


def display_image_with_prediction(model, img):
    resized_img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(resized_img) * 0.8
    img_array = np.clip(img_array, 0, 255)
    
    plt.imshow(img_array.astype(np.uint8))
    plt.axis('off')
    st.pyplot(plt)

    predicted_emotion = predict_emotion(model, img)
    st.write(f'Hasil Prediksi: {predicted_emotion}')

# Streamlit app
st.title("Emotion Detector")

# Upload file section
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        img_path = temp_file.name
    
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Sekarang Anda dapat memproses file gambar tanpa perlu menyimpannya ke direktori permanen
    # Memuat model
    model = load_model()
    # Menampilkan gambar dengan prediksi emosi
    display_image_with_prediction(model, img)
else:
    st.write("Please upload an image file.")
