import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
# Load dataset
df = pd.read_csv("../data/aquaculture_data.csv")

# Features and target
X = df[["temperature", "dissolved_oxygen", "pH", "ammonia"]]
y = df["risk"]

# Encode labels (Low=0, Medium=1, High=2)
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))


# after training Random Forest
best_model = models["Random Forest"]
joblib.dump(best_model, "../models/model.pkl")
joblib.dump(le, "../models/label_encoder.pkl")