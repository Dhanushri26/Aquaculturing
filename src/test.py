import pandas as pd

df = pd.read_csv("../data/WQD.csv")

df.columns = [
    "temperature",
    "turbidity",
    "dissolved_oxygen",
    "bod",
    "co2",
    "ph",
    "alkalinity",
    "hardness",
    "calcium",
    "ammonia",
    "nitrite",
    "phosphorus",
    "h2s",
    "plankton",
    "water_quality"
]
df = df[[
    "temperature",
    "dissolved_oxygen",
    "ph",
    "ammonia",
    "water_quality"
]]
df["temperature"] = (df["temperature"] - 32) * 5/9

print(df["temperature"].describe())

lower = df["temperature"].quantile(0.01)
upper = df["temperature"].quantile(0.99)

df = df[(df["temperature"] >= lower) & (df["temperature"] <= upper)]
df["risk"] = df["water_quality"].map({
    0: "Low",
    1: "Medium",
    2: "High"
})

df = df.drop("water_quality", axis=1)

df.to_csv("../data/aquaculture_data.csv", index=False)
