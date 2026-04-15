import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app


BENCHMARK_CASES = [
    {
        "name": "Low Risk Reference",
        "expected": "Low",
        "payload": {
            "temperature": 22.5,
            "dissolved_oxygen": 8.8,
            "ph": 7.2,
            "ammonia": 0.03,
        },
    },
    {
        "name": "Medium Risk Reference",
        "expected": "Medium",
        "payload": {
            "temperature": 28.0,
            "dissolved_oxygen": 5.4,
            "ph": 7.8,
            "ammonia": 0.22,
        },
    },
    {
        "name": "High Risk Reference",
        "expected": "High",
        "payload": {
            "temperature": 33.0,
            "dissolved_oxygen": 3.2,
            "ph": 8.7,
            "ammonia": 1.35,
        },
    },
    {
        "name": "Borderline High Signals",
        "expected": "Medium",
        "payload": {
            "temperature": 31.0,
            "dissolved_oxygen": 4.0,
            "ph": 8.2,
            "ammonia": 1.18,
        },
    },
]


def main():
    client = app.test_client()

    for case in BENCHMARK_CASES:
        response = client.post("/predict", json=case["payload"])
        result = response.get_json()

        print(f"\n{case['name']}")
        print(f"Expected class: {case['expected']}")
        print(f"HTTP status: {response.status_code}")
        print(f"Predicted class: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Status: {result.get('status')}")
        print(f"Suggestion: {result.get('suggestion')}")
        print(f"Model alerts: {result.get('model_alerts')}")
        print(f"Warnings: {result.get('warnings')}")
        print(f"Probabilities: {result.get('probabilities')}")


if __name__ == "__main__":
    main()
