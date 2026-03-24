import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
os.makedirs("models", exist_ok=True)

synthetic_path = os.path.join(BASE_DIR, "data", "aquaculture_data.csv")
real_path = os.path.join(BASE_DIR, "data", "WQD.csv")

synthetic_df = pd.read_csv(synthetic_path)
df = pd.read_csv(real_path)
# ----------------------------
# Load Synthetic Data
# ----------------------------
# synthetic_df = pd.read_csv("data/aquaculture_data.csv")

# # ----------------------------
# # Load Real Dataset
# # ----------------------------
# df = pd.read_csv("data/WQD.csv")

# ----------------------------
# Clean Column Names
# ----------------------------
df.columns = [
    "temperature","turbidity","dissolved_oxygen","bod","co2","ph",
    "alkalinity","hardness","calcium","ammonia","nitrite",
    "phosphorus","h2s","plankton","water_quality"
]

# ----------------------------
# Convert Temperature (F → C)
# ----------------------------
# Only convert if value clearly in Fahrenheit
df["temperature"] = df["temperature"].apply(
    lambda x: (x - 32) * 5/9 if x > 45 else x
)

# ----------------------------
# Remove extreme outliers (quantile)
# ----------------------------
for col in ["temperature", "dissolved_oxygen", "ph", "ammonia"]:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# ----------------------------
# Select Features
# ----------------------------
df = df[["temperature", "dissolved_oxygen", "ph", "ammonia", "water_quality"]]

# ----------------------------
# Convert Labels
# ----------------------------
df["risk"] = df["water_quality"].map({
    0: "High",
    1: "Low",
    2: "Medium"
})
print(df.groupby("water_quality")[[
    "temperature",
    "dissolved_oxygen",
    "ph",
    "ammonia"
]].mean())
df = df.drop("water_quality", axis=1)

# ----------------------------
# Merge Datasets
# ----------------------------
final_df = pd.concat([synthetic_df, df], ignore_index=True)

low_df = final_df[final_df["risk"] == "Low"]
medium_df = final_df[final_df["risk"] == "Medium"]
high_df = final_df[final_df["risk"] == "High"]

low_upsampled = resample(low_df,
                        replace=True,
                        n_samples=len(medium_df),
                        random_state=42)

final_df = pd.concat([medium_df, high_df, low_upsampled])
# ----------------------------
# Prepare Data
# ----------------------------

X = final_df[["temperature", "dissolved_oxygen", "ph", "ammonia"]]
y = final_df["risk"]

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Train Model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(class_weight = "balanced")
model.fit(X_train, y_train)

models_path = os.path.join(BASE_DIR, "models")
os.makedirs(models_path, exist_ok=True)

joblib.dump(model, os.path.join(models_path, "model.pkl"))
joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))
joblib.dump(le, os.path.join(models_path, "label_encoder.pkl"))
# ----------------------------
# Feature Importance (for explanation)
# ----------------------------
print("\nFeature Importance:")
for name, score in zip(X.columns, model.feature_importances_):
    print(f"{name}: {score:.3f}")

print("\nModel trained and saved successfully!")