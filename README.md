# Aquaculture Risk Prediction System

This project predicts aquaculture water-quality risk from four inputs:

- temperature
- dissolved oxygen
- pH
- ammonia

It combines synthetic sensor-style data with a cleaned real dataset, compares multiple machine-learning models, saves the best one, and serves predictions through a Flask app.

## What Changed

The backend training flow is now consistent and reproducible:

- one shared training pipeline
- stratified train/validation/test splits
- automatic model comparison
- saved scaler, label encoder, and model artifacts
- saved `models/training_report.json` with metrics and dataset details
- API responses that include confidence and top feature drivers

## Current Model Result

The latest verified training run selected `RandomForest` as the best model.

- Validation macro F1: `0.9458`
- Validation accuracy: `0.9468`
- Test accuracy: `0.9508`
- Test macro F1: `0.9498`

## Project Structure

```text
Aquaculturing/
|-- app.py
|-- data/
|   |-- aquaculture_data.csv
|   `-- WQD.csv
|-- models/
|   |-- label_encoder.pkl
|   |-- model.pkl
|   |-- scaler.pkl
|   `-- training_report.json
|-- src/
|   |-- data_generation.py
|   |-- model_selection.py
|   |-- predict.py
|   |-- test.py
|   |-- train_model.py
|   `-- training_pipeline.py
|-- static/
|-- templates/
`-- requirements.txt
```

## Training Pipeline

The main backend logic lives in `src/training_pipeline.py`.

1. Load synthetic data from `data/aquaculture_data.csv`
2. Load real data from `data/WQD.csv`
3. Regenerate the synthetic dataset on every training run to prevent stale or corrupted synthetic samples
4. Rename raw columns and normalize temperature units
5. Remove values outside physical bounds, then trim extreme outliers with 1st and 99th percentile bounds
6. Keep the four prediction features and map labels to `High`, `Low`, and `Medium`
7. Merge synthetic and real datasets
8. Rebalance all classes to the same size
9. Split the data into train, validation, and test sets
10. Compare these models:
   - Logistic Regression
   - Random Forest
   - Extra Trees
   - Gradient Boosting
11. Select the best model by validation macro F1
12. Refit the winning model on train plus validation data
13. Save artifacts and a detailed JSON report

## Files To Run

Train and save the best model:

```bash
python src/train_model.py
```

Compare candidate models without overwriting saved artifacts:

```bash
python src/model_selection.py
```

Run the realtime terminal predictor:

```bash
python src/predict.py
```

Run the Flask web app:

```bash
python app.py
```

Run a quick backend smoke test:

```bash
python src/test.py
```

## API

### `POST /predict`

Request body:

```json
{
  "temperature": 28.0,
  "dissolved_oxygen": 5.4,
  "ph": 7.8,
  "ammonia": 0.22
}
```

Response body:

```json
{
  "success": true,
  "prediction": "Medium",
  "confidence": 0.9896,
  "probabilities": {
    "High": 0.0,
    "Low": 0.0104,
    "Medium": 0.9896
  },
  "suggestion": "Conditions need attention. Monitor dissolved oxygen, temperature, and ammonia closely.",
  "warnings": [],
  "top_factors": [
    {
      "feature": "ammonia",
      "importance": 0.5203
    },
    {
      "feature": "dissolved_oxygen",
      "importance": 0.2624
    }
  ],
  "selected_model": "RandomForest"
}
```

### `GET /model-info`

Returns the saved contents of `models/training_report.json`, including:

- dataset counts
- filtering statistics
- model ranking
- confusion matrix
- class metrics
- feature importance

## Notes

- The project now uses the root `models/` directory as the single source of truth for artifacts.
- `src/models/` has been removed to avoid stale model drift.
- The confidence score comes from the selected model's predicted class probabilities.
- `data/aquaculture_data.csv` is regenerated during training so the stored synthetic data stays aligned with the generator code.

## Next Good Improvements

- add unit tests around preprocessing and API validation
- add input range validation in the web form
- persist experiment history instead of only the latest report
- add SHAP or permutation importance for richer explanations
