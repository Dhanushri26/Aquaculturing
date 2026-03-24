# 🌊 Real-Time Aquaculture Risk Prediction System

## 📌 Overview

This project is an AI-driven system designed to predict fish health risk in aquaculture environments using water quality parameters. It simulates real-time sensor data and applies machine learning to classify risk levels and provide actionable recommendations.

The system is built with a focus on **practical deployment**, **lightweight models**, and **interpretability**, making it suitable for resource-constrained environments.

---

## 🚀 Key Features

* 📡 Real-time simulation of environmental data (temperature, dissolved oxygen, pH, ammonia)
* 🤖 Machine learning-based risk classification (Low / Medium / High)
* 🔁 Hybrid dataset approach (synthetic + real-world water quality data)
* 📊 Model comparison (Logistic Regression, Decision Tree, Random Forest)
* 🧠 Explainable predictions using feature importance
* ⚠️ Decision support system with actionable recommendations
* ⚡ Designed for edge deployment scenarios

---

## 🧠 Problem Statement

Aquaculture systems are highly sensitive to environmental changes such as oxygen depletion, temperature rise, and ammonia accumulation. These changes often go unnoticed until damage occurs.

This project addresses:

> **Early detection of fish mortality risk using data-driven insights**

---

## ⚙️ System Architecture

```
[Simulated / Real Data]
        ↓
[Data Preprocessing]
        ↓
[ML Model (Random Forest)]
        ↓
[Risk Prediction]
        ↓
[Explanation Layer]
        ↓
[Decision Support]
```

---

## 📂 Project Structure

```
aquaculture-ai/
│
├── data/
│   ├── aquaculture_data.csv      # Synthetic dataset
│   └── WQD.csv                  # Real-world dataset
│
├── models/
│   ├── model.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── data_generation.py       # Synthetic data creation
│   ├── train_model.py           # Full training pipeline
│   ├── model_selection.py       # Model comparison (optional)
│   └── predict.py               # Real-time prediction system
│
├── README.md
└── requirements.txt
```

---

## 🧪 Dataset Strategy

This project uses a **hybrid dataset approach**:

### 🔹 Synthetic Data

* Generated using domain-inspired rules
* Ensures controlled learning patterns

### 🔹 Real Dataset (WQD)

* Introduces real-world variability and noise
* Improves generalization

### 🔹 Why Combine Both?

> Synthetic data provides structure, while real data adds realism and variability.

---

## 🤖 Model Selection

Models evaluated:

* Logistic Regression
* Decision Tree
* Random Forest ✅ (Selected)

### ✔ Why Random Forest?

* Handles non-linear relationships
* Robust to noise
* Reduces overfitting compared to single trees
* Performs well on structured/tabular data

---

## 📊 Features Used

* Temperature
* Dissolved Oxygen
* pH
* Ammonia

These features were selected based on their **direct impact on fish survival and water quality**.

---

## ⚡ Real-Time Prediction

The system simulates streaming data and continuously predicts risk:

```
Temp: 24.17°C | DO: 8.67 | pH: 8.42 | NH3: 1.39  
Risk Level: High  
🚨 Immediate action: Increase aeration, reduce feeding  
```

---

## 🧠 Key Learnings

* Importance of data preprocessing and cleaning
* Trade-offs between rule-based systems and ML models
* Handling synthetic vs real-world data
* Model overfitting and generalization
* Designing systems for edge deployment

---

## ⚠️ Limitations

* Uses partially synthetic data
* No real sensor integration
* Simplified environmental modeling

---

## 🔮 Future Improvements

* Integration with IoT sensors for real-time data
* Time-series modeling (LSTM / forecasting)
* Web dashboard for monitoring
* Cloud deployment for scalability
* Advanced anomaly detection

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train Model

```bash
python src/train_model.py
```

### Run Real-Time Prediction

```bash
python src/predict.py
```

---

## 🎯 Conclusion

This project demonstrates how AI can be applied to real-world aquaculture problems by combining domain knowledge, data engineering, and machine learning to build a practical, scalable system.

---

## 📌 Author

**Dhanu Shri V**
