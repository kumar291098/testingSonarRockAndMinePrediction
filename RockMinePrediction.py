# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 22:03:04 2023

@author: Avnish kumar
"""

import numpy as np
#import pandas as pd
# from PIL import Image
import streamlit as st
import pickle

#load model using pickle
lg=pickle.load(open('RockMine.pkl','rb'))
#web app
# img=Image.open('placementImg.jpg')
# st.image(img, width=650)
st.title("SONAR Rock MINE Prediction")
input_text=st.text_input("Enter your all ROCK MINE  feature")

if input_text:
    input_list=input_text.split(',')
    np_df=np.asarray(input_list, dtype=float)
    prediction=lg.predict(np_df.reshape(1, -1))
    if prediction[0]=='R':
        st.write("This is Rock")
    else:
        st.write("This is Mine")
        