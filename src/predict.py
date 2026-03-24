import json
import os
import time

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLUMNS = ["temperature", "dissolved_oxygen", "ph", "ammonia"]
FEATURE_LABELS = {
    "temperature": "Temperature",
    "dissolved_oxygen": "Dissolved Oxygen",
    "ph": "pH",
    "ammonia": "Ammonia",
}

model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

report_path = os.path.join(MODELS_DIR, "training_report.json")
training_report = {}
if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as handle:
        training_report = json.load(handle)


def generate_data():
    temperature = np.random.uniform(20, 35)
    dissolved_oxygen = 10 - (temperature - 20) * 0.3 + np.random.normal(0, 0.5)
    ph = np.random.uniform(6, 9)
    ammonia = np.random.uniform(0.01, 1.5)
    return temperature, dissolved_oxygen, ph, ammonia


def get_suggestion(risk):
    if risk == "High":
        return "Immediate action: increase aeration and reduce feeding."
    if risk == "Medium":
        return "Monitor closely and adjust water conditions."
    return "Conditions look stable."


def get_top_factors():
    return training_report.get("feature_importance", [])[:2]


def get_feature_summary():
    return training_report.get("dataset_summary", {}).get("feature_summary", {})


def get_input_warnings(sample):
    warnings = []
    feature_summary = get_feature_summary()

    for feature in FEATURE_COLUMNS:
        if feature not in feature_summary:
            continue

        value = float(sample.iloc[0][feature])
        bounds = feature_summary[feature]
        if value < bounds["p01"] or value > bounds["p99"]:
            warnings.append(
                f"{FEATURE_LABELS[feature]} outside core range ({bounds['p01']} to {bounds['p99']})"
            )

    return warnings


def get_rule_based_alerts(sample):
    row = sample.iloc[0]
    alerts = []

    if row["dissolved_oxygen"] < 4.0 and row["temperature"] > 28.0:
        alerts.append("Low dissolved oxygen combined with high temperature is a known stress pattern.")
    if row["ammonia"] > 1.2 and row["ph"] > 8.0:
        alerts.append("High ammonia combined with elevated pH can quickly become dangerous.")
    if row["dissolved_oxygen"] < 3.0:
        alerts.append("Dissolved oxygen is critically low.")
    if row["ammonia"] > 1.0:
        alerts.append("Ammonia is approaching a dangerous level.")

    return alerts


while True:
    temperature, dissolved_oxygen, ph, ammonia = generate_data()

    sample = pd.DataFrame(
        [
            {
                "temperature": temperature,
                "dissolved_oxygen": dissolved_oxygen,
                "ph": ph,
                "ammonia": ammonia,
            }
        ]
    )[FEATURE_COLUMNS]

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    probabilities = model.predict_proba(sample_scaled)[0]

    risk = label_encoder.inverse_transform(prediction)[0]
    confidence = float(np.max(probabilities))
    top_factors = get_top_factors()
    warnings = get_input_warnings(sample)
    rule_alerts = get_rule_based_alerts(sample)

    print("\n--- Real-Time Aquaculture Monitor ---")
    print(
        "Temp: "
        f"{temperature:.2f} C | "
        f"DO: {dissolved_oxygen:.2f} | "
        f"pH: {ph:.2f} | "
        f"NH3: {ammonia:.2f}"
    )
    print(f"Risk Level: {risk}")
    print(f"Confidence: {confidence:.2%}")
    print(get_suggestion(risk))
    if warnings:
        print("Warnings: " + "; ".join(warnings))
    if rule_alerts:
        print("Rule alerts: " + "; ".join(rule_alerts))
    if top_factors:
        print(
            "Top model factors: "
            + ", ".join(
                f"{item['feature']} ({item['importance']:.3f})" for item in top_factors
            )
        )

    time.sleep(2)
