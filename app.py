import json
import os
from math import isfinite

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURE_COLUMNS = ["temperature", "dissolved_oxygen", "ph", "ammonia"]
FEATURE_LABELS = {
    "temperature": "Temperature",
    "dissolved_oxygen": "Dissolved Oxygen",
    "ph": "pH",
    "ammonia": "Ammonia",
}
INPUT_LIMITS = {
    "temperature": {"min": 0.0, "max": 40.0},
    "dissolved_oxygen": {"min": 0.0, "max": 20.0},
    "ph": {"min": 0.0, "max": 14.0},
    "ammonia": {"min": 0.0, "max": 5.0},
}

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.json")

model = None
label_encoder = None
scaler = None
training_report = {}


def load_artifacts():
    global model, label_encoder, scaler, training_report

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as handle:
            training_report = json.load(handle)
    else:
        training_report = {}


def build_input_frame(payload):
    values = {}
    errors = []

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    for feature in FEATURE_COLUMNS:
        raw_value = payload.get(feature)
        if raw_value in (None, ""):
            errors.append(f"{FEATURE_LABELS[feature]} is required.")
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{FEATURE_LABELS[feature]} must be a valid number.")
            continue

        if not isfinite(value):
            errors.append(f"{FEATURE_LABELS[feature]} must be finite.")
            continue

        limits = INPUT_LIMITS[feature]
        if value < limits["min"] or value > limits["max"]:
            errors.append(
                f"{FEATURE_LABELS[feature]} must be between {limits['min']} and {limits['max']}."
            )
            continue

        values[feature] = value

    if errors:
        raise ValueError(" ".join(errors))

    return pd.DataFrame([values])[FEATURE_COLUMNS]


def get_top_factors(limit=2):
    return training_report.get("feature_importance", [])[:limit]


def get_feature_summary():
    return training_report.get("dataset_summary", {}).get("feature_summary", {})


def get_input_warnings(input_data):
    warnings = []
    feature_summary = get_feature_summary()

    for feature in FEATURE_COLUMNS:
        if feature not in feature_summary:
            continue

        value = float(input_data.iloc[0][feature])
        bounds = feature_summary[feature]
        if value < bounds["p01"] or value > bounds["p99"]:
            warnings.append(
                f"{FEATURE_LABELS[feature]} is outside the core training range "
                f"({bounds['p01']} to {bounds['p99']})."
            )

    return warnings


def get_suggestion(predicted_label, confidence):
    if predicted_label == "High":
        return "Take immediate action: increase aeration, reduce feeding, and inspect water quality drivers."
    if predicted_label == "Medium":
        if confidence < 0.70:
            return "This is a borderline result. Recheck the inputs and monitor the pond closely."
        return "Conditions need attention. Monitor dissolved oxygen, temperature, and ammonia closely."
    if confidence < 0.70:
        return "The model sees low risk, but confidence is limited. Recheck the readings before relying on it."
    return "Conditions appear stable based on the current readings."


def format_probabilities(probabilities):
    probability_map = {}
    for label, value in zip(label_encoder.classes_, probabilities):
        probability_map[label] = round(float(value), 4)
    return probability_map


def get_rule_based_alerts(input_data):
    row = input_data.iloc[0]
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


def get_response_status(predicted_label, confidence, warnings, rule_alerts):
    if predicted_label == "High":
        return "critical"
    if rule_alerts:
        return "watch"
    if warnings or confidence < 0.70:
        return "caution"
    if predicted_label == "Medium":
        return "watch"
    return "stable"


load_error = None
try:
    load_artifacts()
    print("Model artifacts loaded successfully.")
except Exception as exc:
    load_error = str(exc)
    print(f"Error loading model artifacts: {exc}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or label_encoder is None or scaler is None:
        return jsonify({"error": load_error or "Model artifacts not loaded."}), 500

    try:
        payload = request.get_json(force=True)
        input_data = build_input_frame(payload)
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)[0]
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        confidence = float(probabilities.max())
        warnings = get_input_warnings(input_data)
        rule_alerts = get_rule_based_alerts(input_data)

        return jsonify(
            {
                "success": True,
                "prediction": predicted_label,
                "confidence": round(confidence, 4),
                "probabilities": format_probabilities(probabilities),
                "suggestion": get_suggestion(predicted_label, confidence),
                "warnings": warnings,
                "rule_alerts": rule_alerts,
                "status": get_response_status(predicted_label, confidence, warnings, rule_alerts),
                "top_factors": get_top_factors(),
                "selected_model": training_report.get("selected_model"),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/model-info", methods=["GET"])
def model_info():
    if not training_report:
        return jsonify({"error": "Training report not available."}), 404
    return jsonify(training_report)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
