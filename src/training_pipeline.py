"""
Aquaculture Water Quality Risk Classifier — Improved Training Pipeline
=======================================================================
Key improvements over the original:
  - Stratified K-Fold cross-validation instead of a single lucky/unlucky split
  - Optuna hyperparameter tuning for every candidate model
  - Calibrated probability outputs (CalibratedClassifierCV)
  - Rolling-window & delta features for time-aware training
  - ONNX export for low-latency production inference
  - Evidently-compatible feature-distribution snapshot for drift detection
  - Sensor fault / anomaly pre-filter using Isolation Forest
  - Confidence-threshold metadata saved alongside the model
  - Full data-lineage and reproducibility metadata in the training report
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

# ---------------------------------------------------------------------------
# Optional ONNX export — gracefully skipped if libraries are absent
# ---------------------------------------------------------------------------
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    _ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ONNX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_FEATURES = ["temperature", "dissolved_oxygen", "ph", "ammonia"]
LABEL_COLUMN = "risk"

# Rolling-window and delta features are derived at runtime; all features that
# the final model actually sees are collected after feature engineering.
ENGINEERED_FEATURES: list[str] = []  # filled by engineer_features()

REAL_LABEL_MAP = {0: "High", 1: "Low", 2: "Medium"}

PHYSICAL_LIMITS: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 40.0),
    "dissolved_oxygen": (0.0, 20.0),
    "ph": (0.0, 14.0),
    "ammonia": (0.0, 5.0),
}

RANDOM_STATE = 42
N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 30          # increase for a production run
CONFIDENCE_THRESHOLD = 0.60   # predictions below this are flagged as uncertain

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# 1.  DATA LOADING
# ===========================================================================

def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (synthetic_df, real_df).  Synthetic data is generated on-the-fly."""
    from data_generation import generate_synthetic_data  # local module

    synthetic_df = generate_synthetic_data(n_samples=4200, random_state=RANDOM_STATE)
    synthetic_df.to_csv(DATA_DIR / "aquaculture_data.csv", index=False)

    real_df = pd.read_csv(DATA_DIR / "WQD.csv")
    return synthetic_df, real_df


# ===========================================================================
# 2.  REAL-DATASET CLEANING
# ===========================================================================

def clean_real_dataset(real_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = real_df.copy()
    df.columns = [
        "temperature", "turbidity", "dissolved_oxygen", "bod", "co2",
        "ph", "alkalinity", "hardness", "calcium", "ammonia",
        "nitrite", "phosphorus", "h2s", "plankton", "water_quality",
    ]

    # Fahrenheit → Celsius
    df["temperature"] = df["temperature"].apply(
        lambda v: (v - 32) * 5 / 9 if v > 45 else v
    )

    physical_stats: list[dict] = []
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        before = len(df)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
        physical_stats.append({
            "feature": col, "min_allowed": lo, "max_allowed": hi,
            "rows_removed": before - len(df),
        })

    quantile_stats: list[dict] = []
    for col in RAW_FEATURES:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        before = len(df)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
        quantile_stats.append({
            "feature": col,
            "lower_quantile": round(float(lo), 4),
            "upper_quantile": round(float(hi), 4),
            "rows_removed": before - len(df),
        })

    df = df[RAW_FEATURES + ["water_quality"]].copy()
    df[LABEL_COLUMN] = df["water_quality"].map(REAL_LABEL_MAP)
    df.drop(columns=["water_quality"], inplace=True)

    summary = {
        "rows_before": int(len(real_df)),
        "rows_after": int(len(df)),
        "class_counts": df[LABEL_COLUMN].value_counts().sort_index().to_dict(),
        "physical_limits": physical_stats,
        "quantile_filters": quantile_stats,
    }
    return df, summary


# ===========================================================================
# 3.  ANOMALY PRE-FILTER  (Isolation Forest)
# ===========================================================================

def fit_anomaly_filter(X: pd.DataFrame, contamination: float = 0.03) -> IsolationForest:
    """
    Fit an Isolation Forest on training features.
    In production, run new sensor readings through this first;
    flag (don't discard) anomalies before passing to the risk classifier.
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X)
    return iso


# ===========================================================================
# 4.  FEATURE ENGINEERING  (time-aware rolling / delta features)
# ===========================================================================

def engineer_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling statistics and delta (rate-of-change) features.

    In a real-time system the feature service (Redis / Feast) maintains the
    per-sensor rolling buffer and computes these before calling the model.
    Here we simulate that on the static dataset by treating the DataFrame rows
    as an ordered time series (sorted by index).

    New columns added per raw feature F:
        F_roll_mean  — rolling mean over `window` rows
        F_roll_std   — rolling std
        F_delta      — first-order difference (current − previous)
    """
    out = df.copy()
    for col in RAW_FEATURES:
        out[f"{col}_roll_mean"] = (
            out[col].rolling(window, min_periods=1).mean()
        )
        out[f"{col}_roll_std"] = (
            out[col].rolling(window, min_periods=1).std().fillna(0.0)
        )
        out[f"{col}_delta"] = out[col].diff().fillna(0.0)

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (raw + engineered) present in the DataFrame."""
    return [c for c in df.columns if c != LABEL_COLUMN]


# ===========================================================================
# 5.  DATASET BALANCING
# ===========================================================================

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    target = int(df[LABEL_COLUMN].value_counts().max())
    parts = []
    for label in sorted(df[LABEL_COLUMN].unique()):
        subset = df[df[LABEL_COLUMN] == label]
        if len(subset) < target:
            subset = resample(subset, replace=True, n_samples=target,
                              random_state=RANDOM_STATE)
        parts.append(subset)

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


# ===========================================================================
# 6.  FULL DATA-PREPARATION PIPELINE
# ===========================================================================

def prepare_training_dataframe() -> tuple[pd.DataFrame, list[str], dict]:
    synthetic_df, real_df = load_source_data()
    cleaned_real_df, real_summary = clean_real_dataset(real_df)

    merged = pd.concat([synthetic_df, cleaned_real_df], ignore_index=True)
    balanced = balance_dataset(merged)

    # Feature engineering — must happen AFTER balancing so rolling stats
    # don't leak across the original/synthetic boundary in a misleading way.
    balanced = engineer_features(balanced)
    feature_cols = get_feature_columns(balanced)

    # Save the list globally so other helpers can reference it.
    global ENGINEERED_FEATURES
    ENGINEERED_FEATURES = feature_cols

    summary = {
        "synthetic_rows": int(len(synthetic_df)),
        "real_rows_raw": int(len(real_df)),
        "real_rows_cleaned": int(len(cleaned_real_df)),
        "merged_class_counts_before_balancing": (
            merged[LABEL_COLUMN].value_counts().sort_index().to_dict()
        ),
        "balanced_class_counts": (
            balanced[LABEL_COLUMN].value_counts().sort_index().to_dict()
        ),
        "real_data_cleaning": real_summary,
        "feature_columns": feature_cols,
        "feature_summary": {
            f: {
                "min": round(float(balanced[f].min()), 4),
                "max": round(float(balanced[f].max()), 4),
                "mean": round(float(balanced[f].mean()), 4),
                "p01": round(float(balanced[f].quantile(0.01)), 4),
                "p99": round(float(balanced[f].quantile(0.99)), 4),
            }
            for f in feature_cols
        },
    }
    return balanced, feature_cols, summary


# ===========================================================================
# 7.  OPTUNA HYPERPARAMETER SEARCH
# ===========================================================================

def _optuna_objective(
    trial: optuna.Trial,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Maximise macro-F1 via cross-validation for a given model family."""
    if model_name == "LogisticRegression":
        model = LogisticRegression(
            C=trial.suggest_float("C", 0.01, 10.0, log=True),
            max_iter=trial.suggest_int("max_iter", 500, 5000, step=500),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif model_name == "ExtraTrees":
        model = ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds, average="macro",
                                zero_division=0))
    return float(np.mean(scores))


def tune_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
) -> tuple[optuna.Study, dict]:
    study = optuna.create_study(
        direction="maximize",
        study_name=model_name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, model_name, X, y),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study, study.best_params


# ===========================================================================
# 8.  MODEL CONSTRUCTION FROM BEST PARAMS
# ===========================================================================

def build_model_from_params(model_name: str, params: dict) -> Any:
    if model_name == "LogisticRegression":
        return LogisticRegression(
            **params, class_weight="balanced", random_state=RANDOM_STATE
        )
    if model_name == "RandomForest":
        return RandomForestClassifier(
            **params, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=1,
        )
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            **params, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1,
        )
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {model_name}")


# ===========================================================================
# 9.  EVALUATION HELPERS
# ===========================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    labels = list(label_encoder.classes_)
    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(
            f1_score(y_true, y_pred, average="weighted")), 4),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def build_feature_importance(model: Any, feature_names: list[str]) -> list[dict]:
    # Unwrap CalibratedClassifierCV to reach the underlying estimator
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator

    values = None
    if hasattr(base, "feature_importances_"):
        values = base.feature_importances_
    elif hasattr(base, "coef_"):
        values = np.abs(base.coef_).mean(axis=0)

    if values is None:
        return []

    return sorted(
        [{"feature": f, "importance": round(float(v), 4)}
         for f, v in zip(feature_names, values)],
        key=lambda x: x["importance"],
        reverse=True,
    )


# ===========================================================================
# 10.  DISTRIBUTION SNAPSHOT  (for drift detection)
# ===========================================================================

def compute_distribution_snapshot(X: pd.DataFrame) -> dict:
    """
    Save per-feature statistics that can be compared against production
    distributions using tools like Evidently AI or WhyLogs.
    """
    snapshot: dict = {}
    for col in X.columns:
        arr = X[col].dropna().values
        snapshot[col] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "p05": round(float(np.percentile(arr, 5)), 6),
            "p25": round(float(np.percentile(arr, 25)), 6),
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p75": round(float(np.percentile(arr, 75)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
        }
    return snapshot


# ===========================================================================
# 11.  ONNX EXPORT
# ===========================================================================

def export_onnx(
    pipeline_steps: list[tuple[str, Any]],
    feature_columns: list[str],
    output_path: Path,
) -> bool:
    """
    Export the scaler + classifier as a single ONNX graph.
    Returns True on success, False if skl2onnx is not installed.
    """
    if not _ONNX_AVAILABLE:
        log.warning("skl2onnx not installed — ONNX export skipped.")
        return False

    from sklearn.pipeline import Pipeline as SKPipeline

    sklearn_pipeline = SKPipeline(pipeline_steps)
    n_features = len(feature_columns)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(sklearn_pipeline, initial_types=initial_type,
                                 target_opset=17)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    log.info("ONNX model saved → %s", output_path)
    return True


# ===========================================================================
# 12.  MAIN TRAINING PIPELINE
# ===========================================================================

def run_training_pipeline(save_artifacts: bool = True) -> dict:
    log.info("=== Aquaculture Risk Classifier — Improved Training Pipeline ===")

    # ── 12.1  Prepare data ──────────────────────────────────────────────────
    dataset, feature_cols, dataset_summary = prepare_training_dataframe()

    X_all = dataset[feature_cols]
    y_raw = dataset[LABEL_COLUMN]

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_raw)

    # ── 12.2  Train / test split (test is held out completely) ──────────────
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_all, y_all,
        test_size=0.15,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )

    # ── 12.3  Scale on dev set ──────────────────────────────────────────────
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    # ── 12.4  Anomaly filter (fit on dev features) ──────────────────────────
    anomaly_filter = fit_anomaly_filter(X_dev_scaled)

    # ── 12.5  Optuna tuning + CV ranking ────────────────────────────────────
    model_names = [
        "LogisticRegression",
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
    ]

    ranking_rows: list[dict] = []
    best_params_map: dict[str, dict] = {}

    for name in model_names:
        log.info("Tuning %s …", name)
        study, best_params = tune_model(name, X_dev_scaled, y_dev)
        best_params_map[name] = best_params

        # Re-evaluate best params with full CV to get stable metric estimate
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
        cv_f1_scores: list[float] = []
        cv_acc_scores: list[float] = []

        base_model = build_model_from_params(name, best_params)
        for train_idx, val_idx in cv.split(X_dev_scaled, y_dev):
            m = clone(base_model)
            m.fit(X_dev_scaled[train_idx], y_dev[train_idx])
            preds = m.predict(X_dev_scaled[val_idx])
            cv_f1_scores.append(
                f1_score(y_dev[val_idx], preds, average="macro", zero_division=0)
            )
            cv_acc_scores.append(accuracy_score(y_dev[val_idx], preds))

        ranking_rows.append({
            "model": name,
            "cv_macro_f1_mean": round(float(np.mean(cv_f1_scores)), 4),
            "cv_macro_f1_std": round(float(np.std(cv_f1_scores)), 4),
            "cv_accuracy_mean": round(float(np.mean(cv_acc_scores)), 4),
            "best_hyperparams": best_params,
            "optuna_best_trial_value": round(study.best_value, 4),
        })
        log.info(
            "  %s — CV macro-F1: %.4f ± %.4f",
            name,
            ranking_rows[-1]["cv_macro_f1_mean"],
            ranking_rows[-1]["cv_macro_f1_std"],
        )

    ranking_rows.sort(
        key=lambda r: (r["cv_macro_f1_mean"], r["cv_accuracy_mean"]),
        reverse=True,
    )

    # ── 12.6  Train final model with calibration ─────────────────────────────
    best_name = ranking_rows[0]["model"]
    log.info("Selected model: %s", best_name)

    base_final = build_model_from_params(best_name, best_params_map[best_name])

    # Calibrated classifier wraps the base model; sigmoid calibration works
    # well for most tree and linear classifiers.
    calibrated_model = CalibratedClassifierCV(
        estimator=base_final,
        method="sigmoid",
        cv=3,
    )
    calibrated_model.fit(X_dev_scaled, y_dev)

    # ── 12.7  Test-set evaluation ─────────────────────────────────────────
    test_preds = calibrated_model.predict(X_test_scaled)
    test_proba = calibrated_model.predict_proba(X_test_scaled)
    test_confidence = test_proba.max(axis=1)
    uncertain_fraction = float((test_confidence < CONFIDENCE_THRESHOLD).mean())

    test_metrics = evaluate_predictions(y_test, test_preds, label_encoder)
    test_metrics["uncertain_fraction"] = round(uncertain_fraction, 4)
    test_metrics["confidence_threshold_used"] = CONFIDENCE_THRESHOLD

    # ── 12.8  Distribution snapshot for drift monitoring ─────────────────
    train_distribution_snapshot = compute_distribution_snapshot(
        pd.DataFrame(X_dev_scaled, columns=feature_cols)
    )

    # ── 12.9  Assemble report ─────────────────────────────────────────────
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "2.0.0",
        "selected_model": best_name,
        "feature_columns": feature_cols,
        "label_classes": list(label_encoder.classes_),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "dataset_summary": dataset_summary,
        "split_summary": {
            "dev_rows": int(len(X_dev)),
            "test_rows": int(len(X_test)),
            "cv_folds": N_CV_FOLDS,
        },
        "model_ranking": ranking_rows,
        "test_metrics": test_metrics,
        "feature_importance": build_feature_importance(
            calibrated_model, feature_cols
        ),
        "train_distribution_snapshot": train_distribution_snapshot,
        "onnx_exported": False,
    }

    # ── 12.10  Save artifacts ─────────────────────────────────────────────
    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(calibrated_model, MODELS_DIR / "model.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
        joblib.dump(anomaly_filter, MODELS_DIR / "anomaly_filter.pkl")

        with open(MODELS_DIR / "training_report.json", "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

        # ONNX export — scaler + calibrated model as a single graph
        onnx_path = MODELS_DIR / "model.onnx"
        from sklearn.pipeline import Pipeline as SKPipeline
        onnx_success = export_onnx(
            pipeline_steps=[
                ("scaler", scaler),
                ("classifier", calibrated_model),
            ],
            feature_columns=feature_cols,
            output_path=onnx_path,
        )
        report["onnx_exported"] = onnx_success

        log.info("Artifacts saved to %s", MODELS_DIR)

    log.info(
        "Done — test macro-F1: %.4f  |  accuracy: %.4f",
        test_metrics["macro_f1"],
        test_metrics["accuracy"],
    )

    return {
        "model": calibrated_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "anomaly_filter": anomaly_filter,
        "report": report,
    }


# ===========================================================================
# 13.  PRODUCTION INFERENCE HELPER
# ===========================================================================

def predict_risk(
    readings: dict[str, float],
    model: Any,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    anomaly_filter: IsolationForest | None = None,
    feature_columns: list[str] | None = None,
) -> dict:
    """
    Single-sample inference entry-point for the FastAPI /predict endpoint.

    Parameters
    ----------
    readings : dict
        Raw sensor values, e.g.
        {"temperature": 28.1, "dissolved_oxygen": 6.4, "ph": 7.2, "ammonia": 0.3}
        Rolling / delta features default to 0.0 if absent (first reading of a session).

    Returns
    -------
    dict with keys:
        predicted_label   : str  — "High" | "Medium" | "Low"
        probabilities     : dict — per-class probabilities
        confidence        : float
        uncertain         : bool — True if max prob < CONFIDENCE_THRESHOLD
        anomaly_detected  : bool — True if Isolation Forest flags the reading
    """
    cols = feature_columns or ENGINEERED_FEATURES or RAW_FEATURES

    row = {col: readings.get(col, 0.0) for col in cols}
    X_raw = pd.DataFrame([row], columns=cols)
    X_scaled = scaler.transform(X_raw)

    anomaly_detected = False
    if anomaly_filter is not None:
        anomaly_detected = bool(anomaly_filter.predict(X_scaled)[0] == -1)

    proba = model.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    return {
        "predicted_label": pred_label,
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(label_encoder.classes_, proba)
        },
        "confidence": round(confidence, 4),
        "uncertain": confidence < CONFIDENCE_THRESHOLD,
        "anomaly_detected": anomaly_detected,
    }


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    results = run_training_pipeline(save_artifacts=True)
    print(json.dumps(
        {k: v for k, v in results["report"].items()
         if k != "train_distribution_snapshot"},
        indent=2,
        default=str,
    ))