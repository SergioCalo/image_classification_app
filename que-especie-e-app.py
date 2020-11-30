# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:50 2020

@author: ASUS
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (75,75)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('my_model.h5')

st.write("""
         # Queres saber de que especie se trata?
         """
         )

st.write("Elixe a imaxe do organismo e descúbreo!")

file = st.file_uploader("Escolle aquí a imaxe", type=["jpg", "png"])
#
if file is None:
    st.text("")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Canis lupus")
    elif np.argmax(prediction) == 1:
        st.write("Canis lupus")
    else:
        st.write("Canis lupus")
    
    st.text("Probability (0: Canis lupus, 1: Canis lupus, 2: Canis lupus)")
    st.write(prediction)
