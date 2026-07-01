"""
Customer Churn Prediction - Flask Web Application
==================================================
Author: ML Engineer
Description: Production-ready Flask app for predicting customer churn
             using a pre-trained machine learning model.
"""

import os
import logging
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App Initialization ───────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "churn-prediction-secret-2024")

# ─── Model & Scaler Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ─── Load Artifacts ───────────────────────────────────────────────────────────
try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("✅  Model and scaler loaded successfully.")
except FileNotFoundError as exc:
    logger.error("❌  Artifact not found: %s", exc)
    model  = None
    scaler = None


# ─── Feature Metadata ─────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def parse_and_validate(form: dict) -> tuple[np.ndarray, list]:
    """
    Parse raw form data, validate it, and return a feature vector.

    Parameters
    ----------
    form : dict
        Raw form data from Flask request.

    Returns
    -------
    features : np.ndarray  shape (1, n_features)
    errors   : list[str]
    """
    errors: list[str] = []

    # ── Numeric fields ──────────────────────────────────────────────────────
    def get_float(key: str, label: str, min_val: float = 0.0) -> float | None:
        raw = form.get(key, "").strip()
        if not raw:
            errors.append(f"{label} is required.")
            return None
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return None
        if val < min_val:
            errors.append(f"{label} must be ≥ {min_val}.")
            return None
        return val

    tenure         = get_float("tenure",         "Tenure",          0)
    monthly_charges = get_float("MonthlyCharges", "Monthly Charges", 0)
    total_charges   = get_float("TotalCharges",   "Total Charges",   0)

    # ── Binary / categorical fields ─────────────────────────────────────────
    def get_int_choice(key: str, label: str, choices: list) -> int | None:
        raw = form.get(key, "").strip()
        if raw == "":
            errors.append(f"{label} is required.")
            return None
        try:
            val = int(raw)
        except ValueError:
            errors.append(f"{label} has an invalid value.")
            return None
        if val not in choices:
            errors.append(f"{label} must be one of {choices}.")
            return None
        return val

    gender           = get_int_choice("gender",           "Gender",             [0, 1])
    senior_citizen   = get_int_choice("SeniorCitizen",    "Senior Citizen",     [0, 1])
    partner          = get_int_choice("Partner",          "Partner",            [0, 1])
    dependents       = get_int_choice("Dependents",       "Dependents",         [0, 1])
    phone_service    = get_int_choice("PhoneService",     "Phone Service",      [0, 1])
    multiple_lines   = get_int_choice("MultipleLines",    "Multiple Lines",     [0, 1, 2])
    internet_service = get_int_choice("InternetService",  "Internet Service",   [0, 1, 2])
    online_security  = get_int_choice("OnlineSecurity",   "Online Security",    [0, 1, 2])
    online_backup    = get_int_choice("OnlineBackup",     "Online Backup",      [0, 1, 2])
    device_protection= get_int_choice("DeviceProtection", "Device Protection",  [0, 1, 2])
    tech_support     = get_int_choice("TechSupport",      "Tech Support",       [0, 1, 2])
    streaming_tv     = get_int_choice("StreamingTV",      "Streaming TV",       [0, 1, 2])
    streaming_movies = get_int_choice("StreamingMovies",  "Streaming Movies",   [0, 1, 2])
    contract         = get_int_choice("Contract",         "Contract",           [0, 1, 2])
    paperless        = get_int_choice("PaperlessBilling", "Paperless Billing",  [0, 1])
    payment_method   = get_int_choice("PaymentMethod",    "Payment Method",     [0, 1, 2, 3])

    if errors:
        return None, errors

    features = np.array([[
        tenure, monthly_charges, total_charges,
        gender, senior_citizen, partner, dependents,
        phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless, payment_method,
    ]], dtype=float)

    return features, []


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Render the home page."""
    return render_template("index.html")

@app.route("/features", methods=["GET"])
def features_page():
    """Render the features page."""
    return render_template("features.html")

@app.route("/about", methods=["GET"])
def about_page():
    """Render the about page."""
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """
    Handle prediction request or render the predict page.
    """
    if request.method == "GET":
        return render_template("predict.html")

    """
    Handle prediction request.

    Accepts both HTML form submissions and JSON (AJAX) requests.
    Returns JSON with prediction result, probability, and any errors.
    """
    if model is None or scaler is None:
        return jsonify({
            "success": False,
            "error": "Model artifacts not loaded. Please run the Jupyter notebook first.",
        }), 503

    features, errors = parse_and_validate(request.form)

    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    try:
        # Scale the feature vector
        features_scaled = scaler.transform(features)

        # Prediction
        prediction = int(model.predict(features_scaled)[0])
        proba      = model.predict_proba(features_scaled)[0]

        churn_probability   = round(float(proba[1]) * 100, 2)
        no_churn_probability = round(float(proba[0]) * 100, 2)

        result_label = "Churn" if prediction == 1 else "No Churn"
        risk_level   = (
            "High Risk"   if churn_probability >= 70 else
            "Medium Risk" if churn_probability >= 40 else
            "Low Risk"
        )

        logger.info(
            "Prediction: %s | Churn Prob: %.2f%%",
            result_label, churn_probability,
        )

        return jsonify({
            "success":              True,
            "prediction":           prediction,
            "result_label":         result_label,
            "churn_probability":    churn_probability,
            "no_churn_probability": no_churn_probability,
            "risk_level":           risk_level,
        })

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Prediction error: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint for deployment monitoring."""
    return jsonify({
        "status":       "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    })


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
