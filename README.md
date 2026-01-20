# Churn Prediction Web App

Small Flask app and training script for predicting customer churn.

## Files
- train_model.py — trains models and writes `churn_model.pkl`, `encoders.pkl`, `scaler.pkl`.
- app.py — Flask web app for single and bulk predictions (uses the pickled artifacts).
- data.csv — training dataset (expected by `train_model.py`).
- requirements.txt — Python dependencies.

## Quickstart
1. Create a virtual environment (recommended) and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train the model (produces `churn_model.pkl`, `encoders.pkl`, `scaler.pkl`):

```powershell
python train_model.py
```

3. Run the web app:

```powershell
python app.py
```

Open http://127.0.0.1:5000/ in a browser. Use the form for single predictions or upload an Excel file with the required columns for bulk predictions.

## Bulk template
From the app you can download a sample Excel template (`Download Template`) or run `download_template` route; the app also writes `Churn_template.xlsx` when using the endpoint.

## Notes
- `train_model.py` uses `LabelEncoder` and `StandardScaler` from scikit-learn; saved encoders are per-column LabelEncoders.
- Excel export/import uses `openpyxl`.
