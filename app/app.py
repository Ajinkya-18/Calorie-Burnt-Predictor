import sys 
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.utils import load_model, preprocess_data


st.set_page_config(
    page_title="Calories Burnt Predictor", 
    page_icon="🔥", 
    layout="wide"
)

try: 
    model = load_model('models/trained_RandomForestRegressor.joblib')

except ValueError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Calories Burnt Prediction App")
st.markdown("Enter your workout details to predict the number of calories you have burnt.")

st.sidebar.header("Enter Your Workout Data")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('male', 'female'))
    age = st.sidebar.slider('Age', 10, 90, 25)
    height = st.sidebar.slider('Height (cm)', 120.0, 230.0, 175.0)
    weight = st.sidebar.slider("Weight (kg)", 30.0, 135.0, 70.0)
    duration = st.sidebar.slider('Workout Duration (minutes)', 1.0, 60.0, 15.0)
    heart_rate = st.sidebar.slider('Average Heart Rate (bpm)', 60, 180, 95)
    body_temp = st.sidebar.slider('Body Temperature (°C)', 36.0, 42.0, 39.0, step=0.1)

    data = {'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight, 
            'Duration': duration, 'Heart_Rate': heart_rate, 'Body_Temp': body_temp}
    
    features = pd.DataFrame(data, index=[0])

    return features


input_df = user_input_features()

st.subheader('Your Input Parameters')
st.write(input_df)

if st.button('Predict Calories Burnt'):
    try:
        processed_input = preprocess_data(input_df.copy(), mode='infer')

        prediction = model.predict(processed_input)

        st.success(f'Prediction Complete!')
        st.metric(label="Predicted Calories Burnt", value=f'{prediction[0]:.2f} kcal')

    except ValueError as e:
        st.error(f'An error occurred during preprocessing: {e}')

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")





