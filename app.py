import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model and label encoder
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'label_encoder.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Label Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model/encoder: {e}")
    model = None
    label_encoder = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.json
        
        # Extract features
        temperature = float(data.get('temperature', 0))
        dissolved_oxygen = float(data.get('dissolved_oxygen', 0))
        ph = float(data.get('ph', 0))
        ammonia = float(data.get('ammonia', 0))
        
        # Create a DataFrame for prediction
        input_data = pd.DataFrame([{
            'temperature': temperature,
            'dissolved_oxygen': dissolved_oxygen,
            'ph': ph,
            'ammonia': ammonia
        }])
        
        # Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({
            'success': True,
            'prediction': predicted_label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
