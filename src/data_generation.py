import pandas as pd
import numpy as np

# Number of samples
n_samples = 800

data = []

for _ in range(n_samples):
    # Temperature (20–35°C)
    temp = np.random.uniform(20, 35)

    # Dissolved Oxygen (inverse relation with temperature + noise)
    do = 10 - (temp - 20) * 0.3 + np.random.normal(0, 0.5)
    do = max(2, min(do, 10))  # clamp between 2–10

    # pH (6–9)
    ph = np.random.uniform(6.0, 9.0)

    # Ammonia (0.01–1.5 mg/L)
    ammonia = np.random.uniform(0.01, 1.5)

    # -------- RULE-BASED LABELING --------
    if do < 4 or ammonia > 1.0:
        risk = "High"
    elif (4 <= do < 6) or (0.5 < ammonia <= 1.0):
        risk = "Medium"
    else:
        risk = "Low"

    data.append([temp, do, ph, ammonia, risk])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "temperature", "dissolved_oxygen", "pH", "ammonia", "risk"
])

# Save dataset
df.to_csv("aquaculture_data.csv", index=False)

# Show sample
print(df.head())

# Check class distribution
print("\nClass Distribution:")
print(df["risk"].value_counts())