import streamlit as st
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("ECG Signal Analysis")

# Upload ECG File
uploaded_file = st.file_uploader("Upload ECG Record (.dat)", type=["dat"])
if uploaded_file:
    record_name = uploaded_file.name.split(".")[0]  # Get filename without extension
    st.write(f"### Processing: {record_name}")

    # Load ECG Data
    try:
        record = wfdb.rdrecord(f"mitdb/{record_name}")
        annotation = wfdb.rdann(f"mitdb/{record_name}", 'atr')

        # Extract ECG Signal
        ecg_signal = record.p_signal[:, 0]  # Use first ECG channel

        # Plot ECG Signal
        st.write("### ECG Waveform")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ecg_signal, label="ECG Signal")
        ax.legend()
        st.pyplot(fig)

        # Dummy Feature Extraction (Modify for Real Features)
        mean_val = np.mean(ecg_signal)
        std_val = np.std(ecg_signal)
        max_val = np.max(ecg_signal)
        min_val = np.min(ecg_signal)

        # Create Feature DataFrame
        df = pd.DataFrame([[mean_val, std_val, max_val, min_val]], columns=["Mean", "Std", "Max", "Min"])

        # Load Pre-trained Model (Replace with actual trained model)
        X = np.random.rand(100, 4)  # Dummy dataset
        y = np.random.choice(["Normal", "Arrhythmia"], 100)  # Dummy labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Prediction
        prediction = model.predict(df)
        st.write(f"### Arrhythmia Prediction: 🚑 **{prediction[0]}**")

    except Exception as e:
        st.error(f"Error processing file: {e}")
