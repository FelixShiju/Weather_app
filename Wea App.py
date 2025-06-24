# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:41:56 2025

@author: Felix
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the model
Wea_model = pickle.load(open('Weather_model.sav', 'rb'))

# Prediction function
def weather_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = Wea_model.predict(input_data_reshaped)
    print(prediction)
    
    if prediction[0] == 0:
        return 'Rainy'
    elif prediction[0] == 1:
        return 'Sunny'
    elif prediction[0] == 2:
        return 'Overcast'
    elif prediction[0] == 3:
        return 'Cloudy'
    elif prediction[0] == 4:
        return 'Snowy'
    else:
        return 'Unknown Weather'

# Main Streamlit app
def main():
    st.title('🌦️ Weather Prediction Web App')

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.text_input('🌡️ Temperature(0-50)')
        humidity = st.text_input('💧 Humidity(0-100)')
        wind_speed = st.text_input('💨 Wind Speed (0.0 - 20)')
        precipitation = st.text_input('🌧️ Precipitation (%) (0-100)')
        cloud_cover = st.text_input('☁️ Cloud Cover (overcast: 0, partly cloudy: 1, clear: 2, cloudy: 3)')

    with col2:
        pressure = st.text_input('🌬️ Atmospheric Pressure(~1000)')
        uv_index = st.text_input('☀️ UV Index (0 - 10)')
        visibility = st.text_input('👁️ Visibility (km) (0.0 - 10.0)')
        location = st.text_input('📍 Location (inland: 0, mountain: 1, coastal: 2)')

    result = ''
    if st.button('🔍 Get Weather Prediction'):
        try:
            st.write("Collecting inputs...")
            st.write(f"Temp: {temperature}, Humidity: {humidity}, Wind: {wind_speed}")
            st.write(f"Precip: {precipitation}, Cloud: {cloud_cover}, Pressure: {pressure}")
            st.write(f"UV: {uv_index}, Visibility: {visibility}, Location: {location}")
        
            input_list = [
                float(temperature), float(humidity), float(wind_speed),
                float(precipitation), float(cloud_cover), float(pressure),
                float(uv_index), float(visibility), float(location)
                ]
        
            result = weather_prediction(input_list)

        except ValueError as e:
            st.error(f"⚠️ Input Error: {e}")
            result = "Please enter valid numbers in all fields."

    st.success(f'Prediction: {result}')

if __name__ == '__main__':
    main()
