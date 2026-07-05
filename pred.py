import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Hypertension Prediction App')
st.write('Enter the patient information to predict the likelihood of hypertension and provide your details to record the visit.')

# Input fields for visitor's name and email
visitor_name = st.text_input('Your Name')
visitor_email = st.text_input('Your Email')


# Numerical features used in training
numerical_features = ['age', 'salt_intake', 'stress_score', 'sleep_duration', 'bmi']

# --- Collect inputs (same as you had) ---
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

# --- Map ordinal/binary features to numeric exactly as training did ---
bp_map = {'Normal': 0, 'Prehypertension': 1, 'Hypertension': 2}
exercise_map = {'Low': 0, 'Moderate': 1, 'High': 2}
binary_map_yesno = {'No': 0, 'Yes': 1}
smoking_map = {'Non-Smoker': 0, 'Smoker': 1}

# Scalar numeric values (raw)
raw_values = {
    'age': age,
    'salt_intake': salt_intake,
    'stress_score': stress_score,
    'bp_history': bp_map[bp_history_display],
    'sleep_duration': sleep_duration,
    'bmi': bmi,
    'family_history': binary_map_yesno[family_history_display],
    'exercise_level': exercise_map[exercise_level_display],
    'smoking_status': smoking_map[smoking_status_display],

}

# --- Build base input dataframe filled with zeros for all expected features ---
# Prefer to use model.feature_names_in_ so we guarantee the same columns used at training
if hasattr(model, 'feature_names_in_'):
    expected_features = list(model.feature_names_in_)
else:
    # fallback list — replace with the exact columns you used in training if needed
    expected_features = [
        'age', 'salt_intake', 'stress_score', 'bp_history', 'sleep_duration', 'bmi', 'family_history', 'exercise_level',
        'smoking_status', 'medication_Beta Blocker', 'medication_Diuretic', 'medication_Other',
        'medication_Unknown'
    ]

# Create a 1-row DataFrame with all zeros (float)
input_df = pd.DataFrame([{c: 0.0 for c in expected_features}])

# --- Put raw numeric & mapped categorical single columns into df ---
for k, v in raw_values.items():
    if k in input_df.columns:
        input_df.at[0, k] = v
    else:
        # if the training features used different names (e.g., you had BP_History numeric),
        # try common alternative names or warn the developer
        st.warning(f"Warning: expected column '{k}' not found among model features.")

# --- Medication: set the dummy column(s) if they exist in expected_features ---
# Training used pd.get_dummies(..., drop_first=True), so one medication dummy may be missing.
# We'll create values for any medication column that exists in expected_features.
med_col_name = f"medication_{medication_display}"
# handle exact match of names (your training used spaces e.g., 'medication_beta_blocker')
if med_col_name in input_df.columns:
    input_df.at[0, med_col_name] = 1
else:
    # Try a fallback transformation: sometimes training column names differ in spacing or case
    # Check all expected feature names and set the one that contains medication_display substring
    matched = [c for c in input_df.columns if c.startswith('medication_') and medication_display.replace(' ', '_') in c.replace(' ', '_')]
    if matched:
        input_df.at[0, matched[0]] = 1
    else:
        # If no medication dummy matches, assume the dropped category was the baseline and nothing to set
        # (leave all medication dummies as 0)
        pass

scaled_input = pd.DataFrame(
    scaler.transform(input_df),
    columns=input_df.columns
)

if st.button('Predict', key='predict_button_main'):
    # your code
    if not visitor_name or not visitor_email:
        st.warning("Please enter your Name and Email to proceed with the prediction.")
    else:
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)[:, 1]
        # ---- SHOW OUTPUT ----
        if prediction[0] == 1:
            st.error(
                f"⚠️ High Risk: The model predicts a high likelihood of hypertension.\n"
                f"Probability: {prediction_proba[0]:.2f}"
            )
        else:
            st.success(
                f"✅ Low Risk: The model predicts low likelihood of hypertension.\n"
                f"Probability: {prediction_proba[0]:.2f}"
            )

 