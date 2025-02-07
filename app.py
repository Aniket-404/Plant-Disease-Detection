# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Loading the Model and saving to cache
@st.cache_resource
def load_model(path):
    
    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_base = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(densenet_base.output)
    output = tf.keras.layers.Dense(4, activation='softmax')(x)

    densenet_model = tf.keras.Model(inputs=densenet_base.input, outputs=output)

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))

    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    outputs = tf.keras.layers.average([densenet_output, xception_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    
    return model

# Removing Streamlit Menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Loading the Model
model = load_model('model.h5')

# Title and Description
st.title('Plant Disease Detection')
st.write("Upload a leaf image to check if the plant is healthy or not.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg"])

# If there is an uploaded file, start making predictions
if uploaded_file is not None:
    
    # Display progress and text
    progress = st.text("Processing Image...")
    my_bar = st.progress(0)
    
    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image.resize((700, 400), Image.LANCZOS), width=None)
    my_bar.progress(40)
    
    # Cleaning the image
    image = clean_image(image)
    
    # Making predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(70)
    
    # Making results
    result = make_results(predictions, predictions_arr)
    
    # Remove progress bar
    my_bar.progress(100)
    progress.empty()
    
    # Show the results
    st.write(f"The plant {result['status']} with {result['prediction']} prediction.")
