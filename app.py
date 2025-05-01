import streamlit as st
import pickle
import numpy as np

# Load models (replace with actual paths if different)
try:
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
    heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure the .sav files are in the same directory.")

# Page Config
st.set_page_config(page_title="🧠 Disease Prediction App", layout="wide")

# Sidebar Navigation
st.sidebar.title("🧬 Disease Predictor")
disease = st.sidebar.radio("Select Disease Model", ("🩸 Diabetes", "❤️ Heart", "🧠 Parkinson's"))

# Strip emoji for logic use
clean_disease = disease.split(" ")[1] if len(disease.split(" ")) > 1 else disease

# Main Title
st.title(f"{disease} Prediction")

# Diabetes Prediction
if clean_disease == "Diabetes":
    st.subheader("Enter the following details:")

    pregnancies = st.slider('Pregnancies', 0, 20, 1)
    glucose = st.slider('Glucose Level', 0, 200, 100)
    blood_pressure = st.slider('Blood Pressure', 0, 140, 70)
    skin_thickness = st.slider('Skin Thickness', 0, 100, 20)
    insulin = st.slider('Insulin Level', 0, 900, 100)
    bmi = st.slider('BMI', 0.0, 70.0, 30.0)
    dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.slider('Age', 10, 100, 30)

    if st.button("🩸 Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)
        result = "✅ Diabetic" if prediction[0] == 1 else "🟢 Not Diabetic"
        st.success(f"Prediction: {result}")

# Heart Prediction
elif clean_disease == "Heart":
    st.subheader("Enter the following details:")

    age = st.slider('Age', 20, 100, 50)
    sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.slider('Resting Blood Pressure', 80, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.selectbox('Resting ECG', [0, 1, 2])
    thalach = st.slider('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.slider('ST Depression Induced', 0.0, 6.0, 1.0)
    slope = st.selectbox('Slope of Peak Exercise', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

    if st.button("❤️ Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(input_data)
        result = "💔 Heart Disease Present" if prediction[0] == 1 else "💚 No Heart Disease"
        st.success(f"Prediction: {result}")

# Parkinson's Prediction
elif clean_disease == "Parkinson's":
    st.subheader("Enter the following voice metrics:")

    fo = st.slider('MDVP:Fo(Hz)', 60.0, 300.0, 120.0)
    fhi = st.slider('MDVP:Fhi(Hz)', 60.0, 400.0, 160.0)
    flo = st.slider('MDVP:Flo(Hz)', 60.0, 300.0, 100.0)
    jitter_percent = st.slider('MDVP:Jitter(%)', 0.0, 1.0, 0.01)
    jitter_abs = st.slider('MDVP:Jitter(Abs)', 0.0, 0.02, 0.005)
    rap = st.slider('MDVP:RAP', 0.0, 0.1, 0.01)
    ppq = st.slider('MDVP:PPQ', 0.0, 0.1, 0.01)
    ddp = st.slider('Jitter:DDP', 0.0, 0.1, 0.01)
    shimmer = st.slider('MDVP:Shimmer', 0.0, 1.0, 0.1)
    shimmer_db = st.slider('MDVP:Shimmer(dB)', 0.0, 2.0, 0.5)
    apq3 = st.slider('Shimmer:APQ3', 0.0, 1.0, 0.1)
    apq5 = st.slider('Shimmer:APQ5', 0.0, 1.0, 0.1)
    apq = st.slider('MDVP:APQ', 0.0, 1.0, 0.1)
    dda = st.slider('Shimmer:DDA', 0.0, 1.0, 0.1)
    nhr = st.slider('NHR', 0.0, 1.0, 0.01)
    hnr = st.slider('HNR', 0.0, 50.0, 20.0)
    rpde = st.slider('RPDE', 0.0, 1.0, 0.5)
    dfa = st.slider('DFA', 0.0, 1.0, 0.5)
    spread1 = st.slider('spread1', -10.0, 0.0, -5.0)
    spread2 = st.slider('spread2', 0.0, 1.0, 0.3)
    d2 = st.slider('D2', 0.0, 4.0, 2.0)
    ppe = st.slider('PPE', 0.0, 1.0, 0.3)

    if st.button("🧠 Predict Parkinson's Disease"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                                rpde, dfa, spread1, spread2, d2, ppe]])
        prediction = parkinsons_model.predict(input_data)
        result = "🧠 Parkinson's Disease Detected" if prediction[0] == 1 else "✅ No Parkinson's Disease"
        st.success(f"Prediction: {result}")
