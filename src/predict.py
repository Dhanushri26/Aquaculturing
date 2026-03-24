import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# ----------------------------
# 1. Load Synthetic Data
# ----------------------------
synthetic_df = pd.read_csv("../data/aquaculture_data.csv")

# ----------------------------
# 2. Load Real Dataset
# ----------------------------
df = pd.read_csv("../data/WQD.csv")

# ----------------------------
# 3. Clean Column Names
# ----------------------------
df.columns = [
    "temperature","turbidity","dissolved_oxygen","bod","co2","ph",
    "alkalinity","hardness","calcium","ammonia","nitrite",
    "phosphorus","h2s","plankton","water_quality"
]

df["temperature"] = df["temperature"].apply(lambda x: (x - 32) * 5/9 if x > 40 else x)
df["temperature"] = df["temperature"].clip(lower=20, upper=35)
df["dissolved_oxygen"] = df["dissolved_oxygen"].clip(lower=2.0)
df["ammonia"] = df["ammonia"].clip(lower=0.0)

# ----------------------------
# 5. Remove extreme outliers (quantile)
# ----------------------------
for col in ["temperature", "dissolved_oxygen", "ph", "ammonia"]:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# ----------------------------
# 6. Select Features
# ----------------------------
df = df[["temperature", "dissolved_oxygen", "ph", "ammonia", "water_quality"]]

# ----------------------------
# 7. Convert Labels
# ----------------------------
df["risk"] = df["water_quality"].map({
    0: "Low",
    1: "Medium",
    2: "High"
})

df = df.drop("water_quality", axis=1)

# ----------------------------
# 8. Merge with Synthetic Data
# ----------------------------
final_df = pd.concat([synthetic_df, df], ignore_index=True)

# ----------------------------
# 9. Prepare Data
# ----------------------------
X = final_df[["temperature", "dissolved_oxygen", "ph", "ammonia"]]
y = final_df["risk"]

le = LabelEncoder()
y = le.fit_transform(y)

# ----------------------------
# 10. Train Model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ----------------------------
# 11. Save Model
# ----------------------------
joblib.dump(model, "../models/model.pkl")
joblib.dump(le, "../models/label_encoder.pkl")

print("Model trained and saved successfully!")