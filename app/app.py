import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, feature order
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.title("Alzheimer's Disease Prediction")
st.write("Enter patient details:")
st.info("💡 Fill values carefully. Higher risk is associated with lower MMSE, poor lifestyle, and advanced age.")
# =========================
# Input fields (MATCH DATASET)
# =========================
data = {
    "FunctionalAssessment": st.number_input(
        "Functional Assessment", 0.0, 10.0, 5.0,
        help="Overall ability to perform daily tasks (0 = poor, 10 = excellent)"
    ),

    "ADL": st.number_input(
        "ADL (Daily Living Activities)", 0.0, 10.0, 5.0,
        help="Ability to perform basic activities like eating, bathing, dressing"
    ),

    "MMSE": st.number_input(
        "MMSE Score", 0.0, 30.0, 20.0,
        help="Cognitive test score (24–30 = normal, lower indicates impairment)"
    ),

    "MemoryComplaints": st.selectbox(
        "Memory Complaints", ["NO", "YES"],
        help="0 = No memory issues, 1 = Patient reports memory problems"
    ),

    "BehavioralProblems": st.selectbox(
        "Behavioral Problems", ["NO", "YES"],
        help="0 = No behavioral issues, 1 = Presence of behavioral changes"
    ),

    "BMI": st.number_input(
        "BMI", 10.0, 50.0, 22.0,
        help="Body Mass Index (18.5–24.9 = normal range)"
    ),

    "DietQuality": st.number_input(
        "Diet Quality", 0.0, 10.0, 5.0,
        help="Quality of diet (0 = poor, 10 = very healthy)"
    ),

    "AlcoholConsumption": st.number_input(
        "Alcohol Consumption", 0.0, 10.0, 2.0,
        help="Frequency of alcohol intake (0 = none, 10 = very high)"
    ),

    "PhysicalActivity": st.number_input(
        "Physical Activity", 0.0, 10.0, 5.0,
        help="Level of physical activity (0 = none, 10 = very active)"
    ),

    "SleepQuality": st.number_input(
        "Sleep Quality", 0.0, 10.0, 6.0,
        help="Sleep quality score (higher is better)"
    ),

    "CholesterolHDL": st.number_input(
        "Cholesterol HDL", 20.0, 100.0, 50.0,
        help="Good cholesterol (higher is better, >40 recommended)"
    ),

    "CholesterolTotal": st.number_input(
        "Cholesterol Total", 100.0, 300.0, 180.0,
        help="Total cholesterol (below 200 is desirable)"
    ),

    "SystolicBP": st.number_input(
        "Systolic Blood Pressure", 80, 200, 120,
        help="Upper blood pressure value (120 = normal)"
    ),

    "Age": st.number_input(
        "Age", 1, 120, 60,
        help="Patient age in years"
    ),
}

# Convert to DataFrame
input_df = pd.DataFrame([data])

# Ensure correct feature order
input_df = input_df[feature_columns]

# Apply scaling
input_scaled = scaler.transform(input_df)

# Prediction

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]

    # Optional: probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][1]
        st.write(f"🧠 Risk Probability: {prob:.2f}")

    if prediction == 1:
        st.error("⚠️ High Risk of Alzheimer's")
    else:
        st.success("✅ Low Risk of Alzheimer's")