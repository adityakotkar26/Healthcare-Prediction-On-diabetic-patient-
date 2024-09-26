import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('diabetes_model.sav', 'rb') as f:
    classifier = pickle.load(f)

# Title of the app
st.title('Diabetes Prediction System')

# Input fields for user data
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=300, step=1)
bp = st.number_input('Blood Pressure Level', min_value=0, max_value=200, step=1)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, step=1)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=0.1)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, step=0.01)
age = st.number_input('Age', min_value=0, max_value=120, step=1)

# Button for prediction
if st.button('Predict'):
    # Input data array
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])

    # Make prediction
    prediction = classifier.predict(input_data)

    # Display the result
    if prediction[0] == 1:
        st.write("You are likely to have diabetes.")
    else:
        st.write("You are unlikely to have diabetes.")
