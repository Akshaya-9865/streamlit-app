import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Function to extract RR interval features
def extract_features(ecg_signal):
    peaks, _ = find_peaks(ecg_signal, distance=200)  # Find R-peaks (QRS complex)
    rr_intervals = np.diff(peaks)  # Compute RR intervals

    if len(rr_intervals) == 0:
        return [0, 0, 0, 0]  # Return default if no peaks are detected

    return [
        np.mean(rr_intervals),
        np.std(rr_intervals),
        np.min(rr_intervals),
        np.max(rr_intervals)
    ]


# Load MIT-BIH Arrhythmia Database
record_names = ["100", "101", "102", "103", "104"]  # Add more as needed
X, y = [], []

for rec in record_names:
    try:
        record = wfdb.rdrecord(f"mitdb/{rec}")
        annotation = wfdb.rdann(f"mitdb/{rec}", "atr")

        ecg_signal = record.p_signal[:, 0]  # Extract ECG lead 0
        features = extract_features(ecg_signal)
        X.append(features)
        y.append(annotation.symbol[0])  # Take first annotation symbol (adjust as needed)

        # Visualization - Plot ECG Waveform
        plt.figure(figsize=(10, 4))
        plt.plot(ecg_signal[:1000], label=f"ECG Signal {rec}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.title(f"ECG Waveform from Record {rec}")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Skipping {rec} due to error: {e}")

# Convert to DataFrame
X = pd.DataFrame(X, columns=["Mean RR", "Std RR", "Min RR", "Max RR"])
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test on a new record
new_record = "100"
record = wfdb.rdrecord(f"mitdb/{new_record}")
ecg_signal = record.p_signal[:, 0]

# Extract features for prediction
test_features = pd.DataFrame([extract_features(ecg_signal)], columns=X.columns)
prediction = clf.predict(test_features)
print("\nPredicted Arrhythmia:", prediction)

