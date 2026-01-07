import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Page Config
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# 2. Load data
@st.cache_data
def load_data():
    try:
        # Guna fail encoded yang kau dah save tadi
        return pd.read_csv("StudentsPerformance_final_encoded.csv")
    except FileNotFoundError:
        st.error("Fail 'StudentsPerformance_final_encoded.csv' tidak dijumpai!")
        st.stop()

df = load_data()

# 3. Model & Scaler
@st.cache_resource
def setup_model(data):
    # Kita ambil SEMUA column asalnya untuk training supaya model tak ralat
    X = data.drop(columns=['Pass/Fail'])
    y = data['Pass/Fail']

    scaler = StandardScaler()
    numerical_cols = ['math score', 'reading score', 'writing score']
    
    X_scaled = X.copy()
    X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    # Calculate metrics
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    return model, scaler, X.columns.tolist(), {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

model, scaler, model_columns, metrics = setup_model(df)

# 4. GUI (Hanya Markah Sahaja)
st.title("📊 Student Pass/Fail Predictor")
st.write("Masukkan markah akademik untuk ramalan purata 90+.")

# Display Model Performance Metrics
st.subheader("📈 Model Performance Metrics")
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}", help="% prediction yang betul")
with metric_cols[1]:
    st.metric("Precision", f"{metrics['precision']:.2%}", help="% dari predict LULUS yang betul")
with metric_cols[2]:
    st.metric("Recall", f"{metrics['recall']:.2%}", help="% dari yang betul LULUS, tertangkap")
with metric_cols[3]:
    st.metric("F1 Score", f"{metrics['f1']:.2%}", help="Balance antara Precision & Recall")

with st.form("input_form"):
    st.subheader("Akademik")
    math = st.number_input("Math Score", 0, 100, 70)
    reading = st.number_input("Reading Score", 0, 100, 70)
    writing = st.number_input("Writing Score", 0, 100, 70)
    
    submit = st.form_submit_button("Analyze Student")

if submit:
    # Sediakan input data (semua column lain kita set jadi 0/default)
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Masukkan markah yang user taip
    input_df['math score'] = math
    input_df['reading score'] = reading
    input_df['writing score'] = writing

    # Scaling markah (penting untuk Logistic Regression)
    input_df[['math score', 'reading score', 'writing score']] = scaler.transform(
        input_df[['math score', 'reading score', 'writing score']]
    )

    # Calculate average score
    average = (math + reading + writing) / 3

    # Predict guna MODEL (Inilah tujuan kita buat Logistic Regression)
    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    # Result
    st.divider()
    st.metric("Purata Markah", f"{average:.2f}")
    
    # Guna 'prediction' daripada model, bukan daripada 'average'
    if prediction == 1:
        st.success(f"### Keputusan: LULUS (Berdasarkan Model AI) 🎉")
        st.write(f"**Keyakinan Model:** {prob:.2%} pelajar ini akan mencapai sasaran.")
    else:
        st.error(f"### Keputusan: GAGAL (Berdasarkan Model AI)")
        st.write(f"**Keyakinan Model:** {(1-prob):.2%} pelajar ini berisiko tidak mencapai sasaran.")
                                                                                                                                                                                                            