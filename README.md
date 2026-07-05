# Hypertension Risk Prediction

A machine learning app that predicts the likelihood of hypertension based on lifestyle and health indicators, served through a Streamlit web interface.

## Overview

This project uses a **Random Forest Classifier** trained on a hypertension dataset (1,985 records) to predict whether a person is likely to have hypertension, based on features like age, salt intake, stress score, sleep duration, BMI, and more.

**Model performance (on test set):**
- Accuracy: **97.7%**
- Precision/Recall: ~0.96–1.00 across both classes

## Project Files

| File | Description |
|---|---|
| `app.py` | Streamlit web app for interactive predictions |
| `rf_model.pkl` | Trained Random Forest model |
| `scaler.pkl` | Fitted `StandardScaler` used to scale input features |
| `requirements.txt` | Python dependencies |
| `model.ipynb` | Notebook with data exploration, preprocessing, and model training |

## Features Used

The model was trained on 13 features, in this exact order:

1. `age`
2. `salt_intake` (g/day)
3. `stress_score` (0–10)
4. `bp_history` — encoded: Normal=0, Prehypertension=1, Hypertension=2
5. `sleep_duration` (hours)
6. `bmi`
7. `family_history` — encoded: No=0, Yes=1
8. `exercise_level` — encoded: Low=0, Moderate=1, High=2
9. `smoking_status` — encoded: Non-Smoker=0, Smoker=1
10. `medication_Beta Blocker` (one-hot)
11. `medication_Diuretic` (one-hot)
12. `medication_Other` (one-hot)
13. `medication_Unknown` (one-hot)

> **Note:** `medication_ACE Inhibitor` is the reference/dropped category from one-hot encoding — it's implied when all four medication dummy columns are 0.

**Target:** `has_hypertension` — 0 (No) / 1 (Yes)

## Feature Importance

Based on the trained Random Forest:

| Rank | Feature | Importance |
|---|---|---|
| 1 | bp_history | 0.272 |
| 2 | age | 0.129 |
| 3 | stress_score | 0.103 |
| 4 | bmi | 0.098 |
| 5 | sleep_duration | 0.098 |
| 6 | salt_intake | 0.097 |
| 7 | family_history | 0.084 |
| 8 | smoking_status | 0.074 |
| 9 | exercise_level | 0.016 |
| 10 | medication_Unknown | 0.008 |
| 11 | medication_Beta Blocker | 0.008 |
| 12 | medication_Diuretic | 0.007 |
| 13 | medication_Other | 0.007 |

`bp_history` is by far the strongest predictor, followed by `age` and `stress_score`.

## Setup

1. Clone or download this project folder (make sure `app.py`, `rf_model.pkl`, and `scaler.pkl` are in the same directory).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL Streamlit prints (usually `http://localhost:8501`) in your browser.

## How Prediction Works

1. The user fills in health/lifestyle details through the Streamlit form.
2. Categorical fields are encoded exactly as during training (ordinal maps for `bp_history`, `exercise_level`, etc.; one-hot for `medication`).
3. All 13 features are assembled **in the exact training order** and passed together to `scaler.transform()`.
4. The scaled features are passed to `rf_model.predict()` / `predict_proba()` to get the prediction and confidence score.

> ⚠️ Common pitfall: the scaler and model expect **all 13 columns at once, in the same order used during training** — not just the numeric ones. Passing a subset of columns (e.g. only numeric fields) will raise a `ValueError` about missing feature names.

## Disclaimer

This tool provides an estimate based on a machine learning model and is **not a substitute for professional medical advice**. Please consult a healthcare provider for medical concerns.
