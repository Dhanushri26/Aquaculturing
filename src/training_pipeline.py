import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from data_generation import generate_synthetic_data
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = ["temperature", "dissolved_oxygen", "ph", "ammonia"]
LABEL_COLUMN = "risk"
REAL_LABEL_MAP = {0: "High", 1: "Low", 2: "Medium"}
RANDOM_STATE = 42
PHYSICAL_LIMITS = {
    "temperature": (0.0, 40.0),
    "dissolved_oxygen": (0.0, 20.0),
    "ph": (0.0, 14.0),
    "ammonia": (0.0, 5.0),
}


def load_source_data():
    synthetic_df = generate_synthetic_data(n_samples=4200, random_state=RANDOM_STATE)
    synthetic_df.to_csv(DATA_DIR / "aquaculture_data.csv", index=False)
    real_df = pd.read_csv(DATA_DIR / "WQD.csv")
    return synthetic_df, real_df


def clean_real_dataset(real_df):
    cleaned_df = real_df.copy()
    cleaned_df.columns = [
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
        "water_quality",
    ]

    cleaned_df["temperature"] = cleaned_df["temperature"].apply(
        lambda value: (value - 32) * 5 / 9 if value > 45 else value
    )

    physical_limit_stats = []
    for column, (minimum, maximum) in PHYSICAL_LIMITS.items():
        previous_count = len(cleaned_df)
        cleaned_df = cleaned_df[(cleaned_df[column] >= minimum) & (cleaned_df[column] <= maximum)]
        physical_limit_stats.append(
            {
                "feature": column,
                "min_allowed": minimum,
                "max_allowed": maximum,
                "rows_removed": int(previous_count - len(cleaned_df)),
            }
        )

    filter_stats = []
    for column in FEATURE_COLUMNS:
        lower = cleaned_df[column].quantile(0.01)
        upper = cleaned_df[column].quantile(0.99)
        previous_count = len(cleaned_df)
        cleaned_df = cleaned_df[(cleaned_df[column] >= lower) & (cleaned_df[column] <= upper)]
        filter_stats.append(
            {
                "feature": column,
                "lower_quantile": round(float(lower), 4),
                "upper_quantile": round(float(upper), 4),
                "rows_removed": int(previous_count - len(cleaned_df)),
            }
        )

    cleaned_df = cleaned_df[FEATURE_COLUMNS + ["water_quality"]].copy()
    cleaned_df[LABEL_COLUMN] = cleaned_df["water_quality"].map(REAL_LABEL_MAP)
    cleaned_df = cleaned_df.drop(columns=["water_quality"])

    summary = {
        "rows_before_filtering": int(len(real_df)),
        "rows_after_filtering": int(len(cleaned_df)),
        "class_counts_after_filtering": {
            str(label): int(count)
            for label, count in cleaned_df[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "physical_limits": physical_limit_stats,
        "filters": filter_stats,
    }
    return cleaned_df, summary


def balance_dataset(dataset):
    target_size = int(dataset[LABEL_COLUMN].value_counts().max())
    balanced_parts = []

    for label in sorted(dataset[LABEL_COLUMN].unique()):
        label_df = dataset[dataset[LABEL_COLUMN] == label]
        if len(label_df) < target_size:
            label_df = resample(
                label_df,
                replace=True,
                n_samples=target_size,
                random_state=RANDOM_STATE,
            )
        balanced_parts.append(label_df)

    balanced_df = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    return balanced_df


def prepare_training_dataframe():
    synthetic_df, real_df = load_source_data()
    cleaned_real_df, real_summary = clean_real_dataset(real_df)

    merged_df = pd.concat([synthetic_df, cleaned_real_df], ignore_index=True)
    balanced_df = balance_dataset(merged_df)

    dataset_summary = {
        "synthetic_rows": int(len(synthetic_df)),
        "real_rows_raw": int(len(real_df)),
        "real_rows_cleaned": int(len(cleaned_real_df)),
        "merged_class_counts_before_balancing": {
            str(label): int(count)
            for label, count in merged_df[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "balanced_class_counts": {
            str(label): int(count)
            for label, count in balanced_df[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "real_data_cleaning": real_summary,
        "feature_summary": {
            feature: {
                "min": round(float(balanced_df[feature].min()), 4),
                "max": round(float(balanced_df[feature].max()), 4),
                "mean": round(float(balanced_df[feature].mean()), 4),
                "p01": round(float(balanced_df[feature].quantile(0.01)), 4),
                "p99": round(float(balanced_df[feature].quantile(0.99)), 4),
            }
            for feature in FEATURE_COLUMNS
        },
    }
    return balanced_df, dataset_summary


def build_model_candidates():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=2.0,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def build_feature_importance(model, feature_names):
    values = None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_).mean(axis=0)

    if values is None:
        return []

    ranked = sorted(
        (
            {
                "feature": feature,
                "importance": round(float(score), 4),
            }
            for feature, score in zip(feature_names, values)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked


def _evaluate_predictions(y_true, y_pred, label_encoder):
    labels = list(label_encoder.classes_)
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def _serialize_ranking(ranking_rows):
    serialized = []
    for row in ranking_rows:
        serialized.append(
            {
                "model": row["model"],
                "validation_accuracy": round(float(row["validation_accuracy"]), 4),
                "validation_macro_f1": round(float(row["validation_macro_f1"]), 4),
                "validation_weighted_f1": round(float(row["validation_weighted_f1"]), 4),
            }
        )
    return serialized


def run_training_pipeline(save_artifacts=True):
    dataset, dataset_summary = prepare_training_dataframe()

    X = dataset[FEATURE_COLUMNS]
    y_raw = dataset[LABEL_COLUMN]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    candidates = build_model_candidates()
    ranking_rows = []

    for model_name, estimator in candidates.items():
        model = clone(estimator)
        model.fit(X_train_scaled, y_train)
        val_predictions = model.predict(X_val_scaled)
        metrics = _evaluate_predictions(y_val, val_predictions, label_encoder)
        ranking_rows.append(
            {
                "model": model_name,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_weighted_f1": metrics["weighted_f1"],
            }
        )

    ranking_rows.sort(
        key=lambda row: (
            row["validation_macro_f1"],
            row["validation_accuracy"],
            row["validation_weighted_f1"],
        ),
        reverse=True,
    )

    best_model_name = ranking_rows[0]["model"]
    final_scaler = StandardScaler()
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.Series(list(y_train) + list(y_val))
    X_train_val_scaled = final_scaler.fit_transform(X_train_val)
    X_test_scaled = final_scaler.transform(X_test)

    final_model = clone(candidates[best_model_name])
    final_model.fit(X_train_val_scaled, y_train_val)
    test_predictions = final_model.predict(X_test_scaled)
    test_metrics = _evaluate_predictions(y_test, test_predictions, label_encoder)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "label_classes": list(label_encoder.classes_),
        "dataset_summary": dataset_summary,
        "split_summary": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "model_ranking": _serialize_ranking(ranking_rows),
        "test_metrics": test_metrics,
        "feature_importance": build_feature_importance(final_model, FEATURE_COLUMNS),
    }

    if save_artifacts:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(final_model, MODELS_DIR / "model.pkl")
        joblib.dump(final_scaler, MODELS_DIR / "scaler.pkl")
        joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
        with open(MODELS_DIR / "training_report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    return {
        "model": final_model,
        "scaler": final_scaler,
        "label_encoder": label_encoder,
        "report": report,
    }
