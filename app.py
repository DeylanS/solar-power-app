import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the Trained Model and Scaler
# Use st.cache_resource to load the model only once and store it in cache
@st.cache_resource
def load_model():
    model = joblib.load('solar_power_classifier.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

classifier, scaler = load_model()

# App Title and Description 
st.title('Solar Power Production Predictor')
st.write("""
This app predicts the power production category of a solar plant based on weather and time inputs.
Please provide the input values on the left sidebar.
""")

# Input explanations
with st.expander("What do these features mean?"):
    st.markdown("""
    - **Radiation (W/m²):** The total intensity of sunlight hitting a horizontal surface at the plant's location. Higher values mean more intense sunlight.
    - **Hour of the Day:** The hour on a 24-hour clock (0 for midnight, 13 for 1 PM). This is the most important factor for the daily sun cycle.
    - **Air Temperature (°C):** The ambient air temperature. Very high temperatures can sometimes slightly decrease a solar panel's efficiency.
    - **Day of the Year:** The day number from 1 (Jan 1st) to 365/366 (Dec 31st). This helps the model understand the current season, which affects sun angle and daylight hours.
    - **Sunshine Duration (minutes):** The number of minutes within the past hour that direct sunlight was not blocked by clouds.
    - **Relative Air Humidity (%):** The amount of moisture in the air. High humidity can be associated with haze or cloud cover, reducing solar radiation.
    - **Air Pressure (hPa):** The atmospheric pressure. Clear, sunny days are often associated with high-pressure systems.
    """)

# Sidebar for User Inputs 
st.sidebar.header('Input Features')

def user_input_features():
    # Create sliders and number inputs for all 7 features
    radiation = st.sidebar.slider('Radiation (W/m²)', 0.0, 1000.0, 500.0)
    hour = st.sidebar.slider('Hour of the Day', 0, 23, 12)
    air_temp = st.sidebar.slider('Air Temperature (°C)', -20.0, 50.0, 25.0)
    day_of_year = st.sidebar.slider('Day of the Year', 1, 365, 180)
    sunshine = st.sidebar.slider('Sunshine Duration (minutes)', 0.0, 60.0, 30.0)
    humidity = st.sidebar.slider('Relative Air Humidity (%)', 0, 100, 50)
    pressure = st.sidebar.slider('Air Pressure (hPa)', 950.0, 1050.0, 1010.0)

    # Store the inputs in a dictionary
    data = {
        'Radiation': radiation,
        'Hour': hour,
        'AirTemperature': air_temp,
        'DayOfYear': day_of_year,
        'Sunshine': sunshine,
        'RelativeAirHumidity': humidity,
        'AirPressure': pressure
    }
    # Convert the dictionary to a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display the user's selected input features
st.subheader('Your Input Parameters')
st.write(input_df)

# Prediction Logic
if st.button('Predict Power Category'):
    # Scale the user's input using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # Make a prediction using the loaded model
    prediction = classifier.predict(input_scaled)
    prediction_proba = classifier.predict_proba(input_scaled)

    # Display the result
    st.subheader('Prediction Result')
    st.success(f'The predicted power category is: **{prediction[0]}**')

    st.subheader('Prediction Probability')
    # Create a nice display for probabilities
    proba_df = pd.DataFrame(prediction_proba, columns=classifier.classes_, index=['Probability'])
    st.write(proba_df)