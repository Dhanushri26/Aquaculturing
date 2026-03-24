import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000):
    data = []

    for _ in range(n_samples):
        temp = np.random.uniform(20, 35)

        # DO inversely related to temperature
        do = 10 - (temp - 20) * 0.3 + np.random.normal(0, 0.5)
        do = max(2, min(do, 10))

        ph = np.random.uniform(6.0, 9.0)
        ammonia = np.random.uniform(0.01, 1.5)

        # Improved rule-based labeling (with interactions)
        if (do < 4 and temp > 28) or (ammonia > 1.2 and ph > 8):
            risk = "High"
        elif (4 <= do < 6) or (0.5 < ammonia <= 1.2):
            risk = "Medium"
        else:
            risk = "Low"

        data.append([temp, do, ph, ammonia, risk])

    df = pd.DataFrame(data, columns=[
        "temperature", "dissolved_oxygen", "ph", "ammonia", "risk"
    ])

    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/aquaculture_data.csv", index=False)
    print("Synthetic data generated and saved!")

