import streamlit as st
import pandas as pd
import joblib

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Hypertension Prediction App')
st.write('Enter the patient information to predict the likelihood of hypertension and provide your details to record the visit.')
