import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

model = tf.keras.models.load_model('dog_cat_kimi_classifier.hdf5')

st.write("""
        # Image Classification
        """
        )

file = st.file_uploader("Upload an image", type=['jpeg','jpg'])

st.set_option('deprecation.showfileUploaderEncoding', False)

class_names = ['dog','cat','Kimi !!']

def import_and_predict(image_data, model):
    size = (160, 160)
    image = ImageOps.fit(image_data, size)
    image = np.asarray(image)
    img_reshape = image[np.newaxis,...]
    prediction = model.predict(img_reshape)
    st.subheader(f"Probabilidad de perro {(prediction[0][0]*100):2.2f}%")
    st.subheader(f"Probabilidad de gato {(prediction[0][1]*100):2.2f}%")
    st.subheader(f"Probabilidad de Kimi!! {(prediction[0][2]*100):2.2f}%")
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    prediction = tf.argmax(prediction, 1)

    if prediction.numpy() == 0:
        st.title("La foto es de un perro 🐶")
    elif prediction.numpy() ==1:
        st.title("La foto es de un gato 🐱")
    else:
        st.title("""La foto es de Kimi !!!! 🎇🧨✨🎉🎃🎊🐱‍👤🐱‍🏍""")