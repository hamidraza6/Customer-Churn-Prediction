# 🔮 Customer Churn Prediction

> An end-to-end Machine Learning web application that predicts customer churn in real time using an ensemble of ML models — built with Python, Flask, Scikit-learn, and a modern glassmorphism UI.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.3-black?style=flat-square&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.1-orange?style=flat-square&logo=scikit-learn)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?style=flat-square&logo=bootstrap)

---

## 📌 Project Overview

Customer churn — when a customer stops doing business with a company — costs the telecom industry billions annually. Acquiring a new customer is **5–25× more expensive** than retaining an existing one.

This project builds a **production-ready churn prediction system** that:

![image alt](https://github.com/hamidraza6/Customer-Churn-Prediction/blob/c06069ba8dc5f44fec7c2fd32c482683ec4ba2a7/Screenshot%202026-07-06%20145348.png)

![image alt](https://github.com/hamidraza6/Customer-Churn-Prediction/blob/11004085da7d9e60c2eb2b571e53a8738ca8c642/Screenshot%202026-07-06%20151554.png)

- Ingests customer data (demographics, services, billing)
- Trains and compares 5 ML models
- Serves real-time predictions via a Flask REST API
- Displays churn probability and risk level in a beautiful web UI

---

## 🧠 Business Problem

**Goal:** Given a telecom customer's profile, predict whether they will churn (leave) within the next month.

**Impact:**
- Proactively identify at-risk customers
- Enable targeted retention campaigns
- Reduce revenue loss from customer attrition

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Synthetic Telco Churn Dataset (IBM-style) |
| Rows | 7,043 |
| Features | 19 input + 1 target |
| Target | `Churn` (0 = Stay, 1 = Leave) |
| Churn Rate | ~26% |

**Feature Groups:**
| Category | Features |
|---|---|
| Account | tenure, MonthlyCharges, TotalCharges |
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Contract | Contract, PaperlessBilling, PaymentMethod |

---

## 🤖 Machine Learning Pipeline

```
Raw Data
   │
   ├── Data Cleaning (nulls, duplicates)
   ├── Feature Engineering (ChargesPerMonth, TenureGroup, ServiceCount)
   ├── Encoding (Label Encoding)
   ├── Scaling (StandardScaler)
   │
   ├── Train-Test Split (80/20, stratified)
   │
   ├── Model Training
   │     ├── Logistic Regression
   │     ├── Decision Tree
   │     ├── Random Forest ✅ (Best)
   │     ├── Gradient Boosting
   │     └── XGBoost
   │
   ├── Model Comparison (Accuracy, F1, AUC-ROC)
   ├── GridSearchCV Hyperparameter Tuning
   ├── Stratified K-Fold Cross Validation
   │
   ├── Evaluation
   │     ├── Confusion Matrix
   │     ├── ROC Curve (AUC ≈ 0.98)
   │     ├── Precision-Recall Curve
   │     ├── Classification Report
   │     └── Feature Importance
   │
   └── Save model.pkl + scaler.pkl
```

**Best Model Performance (Random Forest + GridSearchCV):**

| Metric | Score |
|---|---|
| Accuracy | 95.2% |
| Precision | 94.7% |
| Recall | 93.1% |
| F1-Score | 93.9% |
| AUC-ROC | 0.982 |

---

## 🗂 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py                  # Flask application
├── model.pkl               # Trained ML model (generated)
├── scaler.pkl              # Feature scaler (generated)
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment config (Render/Heroku)
├── runtime.txt             # Python version for deployment
├── README.md
├── .gitignore
│
├── notebook/
│   └── train_model.py      # Full training script
│
├── templates/
│   └── index.html          # Main web UI
│
├── static/
│   ├── css/
│   │   └── style.css       # Glassmorphism design system
│   └── images/             # EDA & evaluation plots (generated)
│
├── data/
│   └── telco_churn.csv     # Dataset (generated)
│
└── screenshots/            # App screenshots
```

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python notebook/train_model.py
```
This generates `model.pkl`, `scaler.pkl`, `data/telco_churn.csv`, and all chart images in `static/images/`.

### 5. Run the Flask App
```bash
python app.py
```
Open your browser at **http://localhost:5000**



## 🔌 API Reference

### `POST /predict`
Predict churn for a single customer.

**Form Parameters:** All 19 features (see form fields in `index.html`)

**Response:**
```json
{
  "success": true,
  "prediction": 1,
  "result_label": "Churn",
  "churn_probability": 82.5,
  "no_churn_probability": 17.5,
  "risk_level": "High Risk"
}
```

### `GET /health`
Health-check endpoint for deployment monitoring.

---

## 📈 EDA Visualisations

All charts are auto-generated by `train_model.py` and saved to `static/images/`:

- `churn_distribution.png` — Target class distribution
- `monthly_charges_dist.png` — Charges by churn status
- `correlation_matrix.png` — Feature correlations
- `model_comparison.png` — Model metrics comparison
- `confusion_matrix.png` — Best model confusion matrix
- `roc_curve.png` — ROC curve with AUC
- `precision_recall_curve.png` — Precision-Recall tradeoff
- `feature_importance.png` — Top predictive features

---

## 🔮 Future Improvements

- [ ] Add SHAP explainability dashboard
- [ ] Integrate real Telco dataset (IBM Watson)
- [ ] Add batch prediction (CSV upload)
- [ ] Add customer segmentation clustering
- [ ] Deploy to Render with CI/CD pipeline
- [ ] Add email alert for high-risk customers
- [ ] Implement A/B testing for retention strategies

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask 3.0 |
| ML | Scikit-learn, XGBoost, Joblib |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Frontend | HTML5, CSS3, Bootstrap 5, JavaScript |
| Deployment | Gunicorn, Render |

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/hamidraza6)
- LinkedIn: [Your LinkedIn](www.linkedin.com/in/hamid-raza-data-scientist)
- Email: hamidraza71450@gamil.com



> ⭐ If you found this project helpful, please give it a star on GitHub!
