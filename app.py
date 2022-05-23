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

model = tf.keras.models.load_model('my_model2.hdf5')

st.write("""
        # Image Classification
        """
        )

file = st.file_uploader("Upload an image", type=['jpeg','jpg'])

st.set_option('deprecation.showfileUploaderEncoding', False)

class_names = ['cat','dog']

def import_and_predict(image_data, model):
    data = np.asarray(image)
    size = (160,160)    
    image_2 = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_3 = np.asarray(image_2)
    img = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(75, 75), interpolation=cv2.INTER_CUBIC))
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    prediction = tf.nn.sigmoid(prediction)
    prediction = tf.where(prediction < 0.5, 0, 1)

    if prediction.numpy() == 0:
        st.title("La foto es de un gato 🐱")
    else:
        st.title("La foto es de un perro 🐶")