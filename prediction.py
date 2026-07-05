import streamlit as st
import pandas as pd
import joblib

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Hypertension Prediction App')
st.write('Enter the patient information to predict the likelihood of hypertension and provide your details to record the visit.')

visitor_name = st.text_input('Enter your name')
visitor_email = st.text_input('Enter your email')

num_features = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']

age = st.slider('Age', 18, 90, 45)
salt_intake = st.slider('Salt Intake (grams/day)', 0.0, 20.0, 10.0, 0.1)
stress_score = st.slider('Stress Score (0-10)', 0, 10, 5)
sleep_duration = st.slider('Sleep Duration (hours)', 3.0, 10.0, 7.0, 0.1)
bmi = st.slider('BMI', 15.0, 40.0, 25.0, 0.1)

bp_history_display = st.selectbox('BP History', ['Normal', 'Prehypertension', 'Hypertension'])
family_history_display = st.selectbox('Family History of Hypertension', ['No', 'Yes'])
exercise_level_display = st.selectbox('Exercise Level', ['Low', 'Moderate', 'High'])
smoking_status_display = st.selectbox('Smoking Status', ['Non-Smoker', 'Smoker'])

medication_options = ['Unknown', 'ACE Inhibitor', 'Beta Blocker', 'Diuretic', 'Other']
medication_display = st.selectbox('Medication', medication_options)