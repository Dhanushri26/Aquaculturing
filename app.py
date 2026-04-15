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
    "temperature": (0.0, 40.0),
    "dissolved_oxygen": (0.0, 20.0),
    "ph": (0.0, 14.0),
    "ammonia": (0.0, 5.0),
}

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.json")

model = None
label_encoder = None
scaler = None
training_report = {}
load_error = None


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_artifacts():
    global model, label_encoder, scaler, training_report
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    training_report = _load_json_if_exists(REPORT_PATH)


def build_input_frame(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    values = {}
    errors = []
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

        lower, upper = INPUT_LIMITS[feature]
        if not (lower <= value <= upper):
            errors.append(f"{FEATURE_LABELS[feature]} must be between {lower} and {upper}.")
            continue

        values[feature] = value

    if errors:
        raise ValueError(" ".join(errors))
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def get_feature_summary():
    return training_report.get("dataset_summary", {}).get("feature_summary", {})


def get_top_factors(limit=2):
    return training_report.get("feature_importance", [])[:limit]


def get_input_warnings(input_data):
    warnings = []
    feature_summary = get_feature_summary()

    for feature in FEATURE_COLUMNS:
        bounds = feature_summary.get(feature)
        if not bounds:
            continue
        value = float(input_data.at[0, feature])
        if value < bounds["p01"] or value > bounds["p99"]:
            warnings.append(
                f"{FEATURE_LABELS[feature]} is outside the core training range ({bounds['p01']} to {bounds['p99']})."
            )
    return warnings


def get_model_alerts(probabilities):
    alerts = []
    sorted_probs = sorted((float(value) for value in probabilities), reverse=True)
    top_probability = sorted_probs[0]
    second_probability = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

    if top_probability < 0.70:
        alerts.append("Prediction confidence is moderate. Consider rechecking sensor readings.")
    if top_probability - second_probability < 0.15:
        alerts.append("Top classes are close. The result may be sensitive to small input changes.")
    return alerts


def get_suggestion(predicted_label, confidence):
    if predicted_label == "High":
        return "Take immediate action: increase aeration, reduce feeding, and inspect water quality drivers."
    if predicted_label == "Medium" and confidence < 0.70:
        return "This is a borderline result. Recheck the inputs and monitor the pond closely."
    if predicted_label == "Medium":
        return "Conditions need attention. Monitor dissolved oxygen, temperature, and ammonia closely."
    if confidence < 0.70:
        return "The model sees low risk, but confidence is limited. Recheck the readings before relying on it."
    return "Conditions appear stable based on the current readings."


def format_probabilities(probabilities):
    return {
        label: round(float(value), 4)
        for label, value in zip(label_encoder.classes_, probabilities)
    }


def get_response_status(predicted_label, confidence, warnings):
    if predicted_label == "High":
        return "critical"
    if warnings or confidence < 0.70:
        return "caution"
    if predicted_label == "Medium":
        return "watch"
    return "stable"


def ensure_model_loaded():
    if model is None or label_encoder is None or scaler is None:
        return False
    return True


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
    if not ensure_model_loaded():
        return jsonify({"error": load_error or "Model artifacts not loaded."}), 500

    try:
        payload = request.get_json(force=True)
        input_data = build_input_frame(payload)
        input_scaled = scaler.transform(input_data)

        predicted_code = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        predicted_label = label_encoder.inverse_transform([predicted_code])[0]
        confidence = float(max(probabilities))
        warnings = get_input_warnings(input_data)

        response = {
            "success": True,
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": format_probabilities(probabilities),
            "suggestion": get_suggestion(predicted_label, confidence),
            "warnings": warnings,
            "model_alerts": get_model_alerts(probabilities),
            "status": get_response_status(predicted_label, confidence, warnings),
            "top_factors": get_top_factors(),
            "selected_model": training_report.get("selected_model"),
        }
        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/model-info", methods=["GET"])
def model_info():
    if not training_report:
        return jsonify({"error": "Training report not available."}), 404
    return jsonify(training_report)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
