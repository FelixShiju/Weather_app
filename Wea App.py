# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:41:56 2025
@author: Felix
"""

import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the trained weather prediction model
Wea_model = pickle.load(open('Weather_model.sav', 'rb'))

# Prediction logic
def weather_prediction(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = Wea_model.predict(input_array)
    weather_labels = ['Rainy', 'Sunny', 'Overcast', 'Cloudy', 'Snowy']
    return weather_labels[prediction[0]] if prediction[0] < len(weather_labels) else 'Unknown Weather'

# Streamlit UI
def main():
    st.set_page_config(page_title="Weather Prediction", page_icon="🌦️")
    st.title("🌦️ Weather Prediction Web App")
    st.markdown("Enter current environmental conditions to estimate upcoming weather using a machine learning model.")

    st.markdown("#### 📋 Enter Weather Parameters")
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider(
            "🌡️ Temperature (°C)",
            0, 50, 25,
            help="Ambient air temperature from 0°C to 50°C"
        )
        humidity = st.slider(
            "💧 Humidity (%)",
            0, 100, 60,
            help="Relative humidity percentage from 0% to 100%"
        )
        wind_speed = st.slider(
            "💨 Wind Speed (km/h)",
            0.0, 20.0, 5.0, step=0.1,
            help="Speed of wind at surface level from 0 to 20 km/h"
        )
        precipitation = st.slider(
            "🌧️ Precipitation Probability (%)",
            0, 100, 20,
            help="Chances of rain/snow based on current conditions"
        )
        cloud_cover = st.selectbox(
            "☁️ Cloud Cover",
            options=["0 - Overcast", "1 - Partly Cloudy", "2 - Clear", "3 - Cloudy"],
            help="Sky condition based on cloud density"
        )

    with col2:
        pressure = st.slider(
            "🌬️ Atmospheric Pressure (hPa)",
            900, 1100, 1013,
            help="Typical sea-level pressure ~1013 hPa"
        )
        uv_index = st.slider(
            "☀️ UV Index",
            0, 11, 5,
            help="Sun exposure index: 0 (low) to 11+ (very high)"
        )
        visibility = st.slider(
            "👁️ Visibility (km)",
            0.0, 10.0, 8.0, step=0.1,
            help="Horizontal visibility in kilometers"
        )
        location = st.selectbox(
            "📍 Location Type",
            options=["0 - Inland", "1 - Mountain", "2 - Coastal"],
            help="General region where conditions are measured"
        )

    st.markdown("----")
    if st.button("🔍 Get Weather Prediction"):
        try:
            input_features = [
                float(temperature),
                float(humidity),
                float(wind_speed),
                float(precipitation),
                int(cloud_cover.split(" - ")[0]),
                float(pressure),
                float(uv_index),
                float(visibility),
                int(location.split(" - ")[0])
            ]

            result = weather_prediction(input_features)
            st.success(f"📡 **Predicted Weather:** {result}")

        except Exception as e:
            st.error(f"⚠️ An error occurred while processing inputs: {e}")

if __name__ == "__main__":
    main()
