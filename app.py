import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

model =  load_model('/Users/sohamdatta/Desktop/SML Folder/Veg:Fruits_Classification/Image_classify.keras')

data_cat=[
    'apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon'
]
 
img_height = 180
img_width = 180

st.header('Vegetables & Fruit Classification Using CNN')
img = st.text_input('Enter image name:  ' ,'ginger.jpg')  

image_load = tf.keras.utils.load_img(img, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims (img_arr, 0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

st.image(img)


st.write('Image is classified to be {} wth accuracy of {:0.2f}'.format(data_cat[np.argmax(score)], np.max(score)*100))