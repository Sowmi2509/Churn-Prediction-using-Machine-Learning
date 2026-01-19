from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("churn_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURES = [
    "SeniorCitizen",
    "tenure",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges"
]

# Define which columns are numeric (should NOT be encoded)
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges"]


def _build_field_options():
    """Return a dict mapping feature -> list of (value,label) tuples for dropdowns."""
    options = {}
    for col in FEATURES:
        # For categorical columns that have encoders, use encoder classes_
        if col in encoders and col not in NUMERIC_COLS:
            classes = getattr(encoders[col], "classes_", None)
            if classes is not None:
                options[col] = [(str(c), str(c)) for c in classes]
        # For SeniorCitizen provide explicit Yes/No choices (values are numeric strings)
        elif col == "SeniorCitizen":
            options[col] = [("0", "No"), ("1", "Yes")]
    return options


def _match_label_case_insensitive(val, encoder):
    """Return the encoder's exact class string that matches `val` case-insensitively."""
    if val is None:
        return val
    s = str(val).strip()
    try:
        classes = getattr(encoder, "classes_", None)
        if classes is None:
            return s
        for c in classes:
            if str(c).strip().lower() == s.lower():
                return c
    except Exception:
        return s
    return s


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    churn_class = None
    single_error = None
    # Field options for rendering dropdowns
    field_options = _build_field_options()
    form_values = {}

    if request.method == "POST":
        if "file" in request.files:
            return render_template("index.html", single_error="Please use the bulk upload form for Excel files.")
        
        try:
            # Collect raw form values
            raw = {f: request.form.get(f) for f in FEATURES}
            form_values = raw

            # Ensure all fields present
            if any(v is None or v == "" for v in raw.values()):
                single_error = "Please fill all customer details."
            else:
                df = pd.DataFrame([raw], columns=FEATURES)

                # Cast numeric columns to float
                for nc in NUMERIC_COLS:
                    df[nc] = pd.to_numeric(df[nc], errors="coerce")

                # Check if numeric conversion produced NaN
                if df[NUMERIC_COLS].isnull().any(axis=None):
                    single_error = "Numeric fields must be valid numbers (SeniorCitizen/tenure/MonthlyCharges)."
                else:
                    # Apply encoders ONLY to categorical features (not numeric ones)
                    for col in FEATURES:
                        if col in encoders and col not in NUMERIC_COLS:
                            # Map input labels case-insensitively
                            df[col] = df[col].astype(str).apply(
                                lambda v: _match_label_case_insensitive(v, encoders[col])
                            )
                            df[col] = encoders[col].transform(df[col].astype(str))

                    # Ensure correct column order and scale
                    df_for_scaler = df[FEATURES].astype(float)
                    df_scaled = scaler.transform(df_for_scaler)

                    result = model.predict(df_scaled)[0]

                    if result == 1:
                        prediction = "Customer Will Churn"
                        churn_class = "yes"
                    else:
                        prediction = "Customer Will NOT Churn"
                        churn_class = "no"

        except Exception as e:
            single_error = f"Invalid input values: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        churn_class=churn_class,
        single_error=single_error,
        field_options=field_options,
        form_values=form_values
    )


@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():
    bulk_error = None

    if "file" not in request.files:
        bulk_error = "No file uploaded. Please upload an Excel file."
        return render_template("index.html", bulk_error=bulk_error)

    file = request.files["file"]

    if file.filename == "":
        bulk_error = "No file selected. Please choose an Excel file."
        return render_template("index.html", bulk_error=bulk_error)

    try:
        df = pd.read_excel(file)

        if df.empty:
            bulk_error = "Data not found. Excel file has no rows."
            return render_template("index.html", bulk_error=bulk_error)

        missing_cols = set(FEATURES) - set(df.columns)
        if missing_cols:
            bulk_error = f"Missing columns: {', '.join(missing_cols)}"
            return render_template("index.html", bulk_error=bulk_error)

        original_df = df.copy()

        # Cast numeric columns to float
        for nc in NUMERIC_COLS:
            if nc in df.columns:
                df[nc] = pd.to_numeric(df[nc], errors="coerce")

        if df[NUMERIC_COLS].isnull().any(axis=None):
            bulk_error = "Numeric columns contain invalid values. Check SeniorCitizen/tenure/MonthlyCharges."
            return render_template("index.html", bulk_error=bulk_error)

        # Apply encoders ONLY to categorical columns (not numeric ones)
        for col in FEATURES:
            if col in encoders and col not in NUMERIC_COLS:
                df[col] = df[col].astype(str).apply(
                    lambda v: _match_label_case_insensitive(v, encoders[col])
                )
                df[col] = encoders[col].transform(df[col].astype(str))

        df_scaled = scaler.transform(df[FEATURES].astype(float))
        preds = model.predict(df_scaled)

        original_df["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in preds]

        output_file = "predicted_churn.xlsx"
        original_df.to_excel(output_file, index=False)

        return send_file(output_file, as_attachment=True)

    except Exception as e:
        bulk_error = f"Invalid Excel format or processing error: {str(e)}"
        return render_template("index.html", bulk_error=bulk_error)


@app.route("/download_template")
def download_template():
    # Use correct data types matching the model training
    template = pd.DataFrame({
        "SeniorCitizen": [0, 1],  # 0 = No, 1 = Yes (numeric as trained)
        "tenure": [12, 24],
        "InternetService": ["Fiber optic", "DSL"],
        "OnlineSecurity": ["Yes", "No"],
        "TechSupport": ["No", "Yes"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
        "MonthlyCharges": [70.0, 55.0]
    })

    template.to_excel("Churn_template.xlsx", index=False)
    return send_file("Churn_template.xlsx", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)