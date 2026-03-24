import numpy as np
import pandas as pd
import joblib
import time

# Load model
model = joblib.load("models/model.pkl")
le = joblib.load("models/label_encoder.pkl")

def generate_data():
    temp = np.random.uniform(20, 35)
    do = 10 - (temp - 20)*0.3 + np.random.normal(0, 0.5)
    ph = np.random.uniform(6, 9)
    ammonia = np.random.uniform(0.01, 1.5)

    return temp, do, ph, ammonia

def get_suggestion(risk):
    if risk == "High":
        return "🚨 Immediate action: Increase aeration, reduce feeding"
    elif risk == "Medium":
        return "⚠️ Monitor closely and adjust conditions"
    else:
        return "✅ Conditions stable"

def explain_prediction(sample_df):
    importances = model.feature_importances_
    features = sample_df.columns

    importance_dict = dict(zip(features, importances))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_features[:2]

while True:
    temp, do, ph, ammonia = generate_data()

    sample = pd.DataFrame([{
        "temperature": temp,
        "dissolved_oxygen": do,
        "ph": ph,
        "ammonia": ammonia
    }])

    pred = model.predict(sample)
    risk = le.inverse_transform(pred)[0]

    explanation = explain_prediction(sample)

    print("\n--- Real-Time Aquaculture Monitor ---")
    print(f"Temp: {temp:.2f}°C | DO: {do:.2f} | pH: {ph:.2f} | NH3: {ammonia:.2f}")
    print(f"Risk Level: {risk}")
    print(get_suggestion(risk))
    print(f"Top Factors: {explanation}")

    time.sleep(2)