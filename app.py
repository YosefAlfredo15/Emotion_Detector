import streamlit as st
from PIL import Image
import numpy as np
import h5py  # Import h5py untuk memuat model

# Function to load the model
@st.cache
def load_emotion_model():
    with h5py.File("emotion_model.h5", "r") as f:  # Ganti "your_model_path.h5" dengan path ke model Anda
        model = f.get("model_weights").value  # Sesuaikan dengan nama dataset model Anda dalam file .h5
    return model

# Function to preprocess the image
def preprocess_image(image):
    # Resize image to required dimensions
    resized_image = image.resize((48, 48))
    # Convert image to grayscale
    grayscale_image = resized_image.convert('L')
    # Convert image to array
    image_array = np.array(grayscale_image) / 255.0
    # Expand dimensions to match model input shape
    processed_image = np.expand_dims(image_array, axis=0)
    return processed_image

# Function to predict emotion and display result
def display_emotion_prediction(model, image):
    # Preprocess image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(processed_image)
    # Map prediction to emotion labels
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    # Display result
    st.write(f"Predicted Emotion: {predicted_emotion}")

# Streamlit app
st.title("Emotion Detector")

# Upload file section
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    model = load_emotion_model()
    if st.button('Predict'):
        display_emotion_prediction(model, image)
else:
    st.write("Please upload an image file.")
