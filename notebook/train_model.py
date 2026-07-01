"""
Customer Churn Prediction - Model Training Script
==================================================
Run this script to train the model and save model.pkl + scaler.pkl
to the project root.

Usage:
    python notebook/train_model.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay,
    accuracy_score, f1_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found – skipping.")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
IMAGES_DIR  = os.path.join(ROOT, "static", "images")
MODEL_PATH  = os.path.join(ROOT, "model.pkl")
SCALER_PATH = os.path.join(ROOT, "scaler.pkl")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#60a5fa"]
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#16162a",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#f1f5f9",
    "grid.color":       "#1e1e35",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE / LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 1: Generate Synthetic Telecom Churn Dataset")
print("="*60)

np.random.seed(42)
N = 7043  # Matches real Telco dataset size

gender           = np.random.choice([0, 1], N)
senior_citizen   = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner          = np.random.choice([0, 1], N)
dependents       = np.random.choice([0, 1], N, p=[0.70, 0.30])
tenure           = np.random.randint(0, 73, N)
phone_service    = np.random.choice([0, 1], N, p=[0.10, 0.90])
multiple_lines   = np.random.choice([0, 1, 2], N, p=[0.42, 0.42, 0.16])
internet_service = np.random.choice([0, 1, 2], N, p=[0.34, 0.44, 0.22])
online_security  = np.random.choice([0, 1, 2], N, p=[0.50, 0.29, 0.21])
online_backup    = np.random.choice([0, 1, 2], N, p=[0.44, 0.35, 0.21])
device_protection= np.random.choice([0, 1, 2], N, p=[0.44, 0.34, 0.22])
tech_support     = np.random.choice([0, 1, 2], N, p=[0.49, 0.29, 0.22])
streaming_tv     = np.random.choice([0, 1, 2], N, p=[0.40, 0.38, 0.22])
streaming_movies = np.random.choice([0, 1, 2], N, p=[0.40, 0.39, 0.21])
contract         = np.random.choice([0, 1, 2], N, p=[0.55, 0.21, 0.24])
paperless        = np.random.choice([0, 1], N, p=[0.41, 0.59])
payment_method   = np.random.choice([0, 1, 2, 3], N)

monthly_charges  = np.round(np.random.uniform(18, 119, N), 2)
total_charges    = np.round(tenure * monthly_charges + np.random.normal(0, 50, N), 2)
total_charges    = np.clip(total_charges, 0, None)

# Churn probability: realistic business rules
churn_prob = (
    0.05
    + 0.30 * (contract == 0)
    - 0.15 * (contract == 2)
    + 0.20 * (internet_service == 1)
    - 0.05 * (online_security == 1)
    - 0.05 * (tech_support == 1)
    + 0.10 * (senior_citizen == 1)
    + 0.25 * (tenure < 12)
    - 0.20 * (tenure > 48)
    + 0.10 * (monthly_charges > 80) / 119
    + 0.05 * (paperless == 1)
    + np.random.normal(0, 0.05, N)
)
churn_prob = np.clip(churn_prob, 0.02, 0.98)
churn      = (np.random.rand(N) < churn_prob).astype(int)

df = pd.DataFrame({
    "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
    "Dependents": dependents, "PhoneService": phone_service,
    "MultipleLines": multiple_lines, "InternetService": internet_service,
    "OnlineSecurity": online_security, "OnlineBackup": online_backup,
    "DeviceProtection": device_protection, "TechSupport": tech_support,
    "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
    "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment_method, "Churn": churn,
})

df.to_csv(os.path.join(DATA_DIR, "telco_churn.csv"), index=False)
print(f"Dataset shape : {df.shape}")
print(f"Churn rate    : {df['Churn'].mean()*100:.1f}%")
print(df.describe().round(2))

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: Data Cleaning")
print("="*60)

print(f"Missing values:\n{df.isnull().sum()}")
df.dropna(inplace=True)
print(f"Duplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Shape after cleaning: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 3: Feature Engineering")
print("="*60)

df["ChargesPerMonth"]   = (df["TotalCharges"] / (df["tenure"] + 1)).round(2)
df["TenureGroup"]       = pd.cut(df["tenure"], bins=[-1,12,24,48,72],
                                  labels=[0,1,2,3], right=True).fillna(0).astype(int)
df["ServiceCount"]      = (
    df["PhoneService"] + (df["MultipleLines"] == 1).astype(int) +
    (df["InternetService"] < 2).astype(int) + (df["OnlineSecurity"] == 1).astype(int) +
    (df["OnlineBackup"] == 1).astype(int) + (df["DeviceProtection"] == 1).astype(int) +
    (df["TechSupport"] == 1).astype(int) + (df["StreamingTV"] == 1).astype(int) +
    (df["StreamingMovies"] == 1).astype(int)
)
print("Engineered features: ChargesPerMonth, TenureGroup, ServiceCount")
print(df[["ChargesPerMonth","TenureGroup","ServiceCount"]].describe().round(2))

# ─────────────────────────────────────────────────────────────────────────────
# 4. EDA — PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 4: EDA & Visualisations")
print("="*60)

# 4a — Churn Distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Churn"].value_counts()
bars   = ax.bar(["No Churn", "Churn"], counts.values, color=[PALETTE[1], PALETTE[3]], width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{val:,}", ha="center", va="bottom", fontweight="bold")
ax.set_title("Churn Distribution", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Count"); ax.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "churn_distribution.png"), dpi=150)
plt.close()
print("Saved: churn_distribution.png")

# 4b — Monthly Charges by Churn
fig, ax = plt.subplots(figsize=(7, 4))
for label, color in zip([0, 1], [PALETTE[1], PALETTE[3]]):
    ax.hist(df[df["Churn"]==label]["MonthlyCharges"], bins=30, alpha=0.7,
            label=["No Churn","Churn"][label], color=color, edgecolor="none")
ax.set_title("Monthly Charges Distribution by Churn", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Monthly Charges ($)"); ax.set_ylabel("Frequency")
ax.legend(); ax.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "monthly_charges_dist.png"), dpi=150)
plt.close()
print("Saved: monthly_charges_dist.png")

# 4c — Correlation Matrix
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "correlation_matrix.png"), dpi=150)
plt.close()
print("Saved: correlation_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE SELECTION & SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 5: Feature Selection & Train-Test Split")
print("="*60)

FEATURES = [
    "tenure","MonthlyCharges","TotalCharges",
    "gender","SeniorCitizen","Partner","Dependents",
    "PhoneService","MultipleLines","InternetService",
    "OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
]
X = df[FEATURES]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 6: Training Multiple Models")
print("="*60)

models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, random_state=42, eval_metric="logloss",
        use_label_encoder=False, n_jobs=-1)

results = {}
for name, clf in models.items():
    clf.fit(X_train_s, y_train)
    preds  = clf.predict(X_test_s)
    probas = clf.predict_proba(X_test_s)[:, 1]
    acc    = accuracy_score(y_test, preds)
    f1     = f1_score(y_test, preds)
    auc    = roc_auc_score(y_test, probas)
    results[name] = {"Accuracy": acc, "F1": f1, "AUC-ROC": auc, "model": clf}
    print(f"  {name:25s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 7: Model Comparison")
print("="*60)

names   = list(results.keys())
accs    = [results[n]["Accuracy"] for n in names]
f1s     = [results[n]["F1"]       for n in names]
aucs    = [results[n]["AUC-ROC"]  for n in names]

x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.25
ax.bar(x - w,  accs, w, label="Accuracy", color=PALETTE[0])
ax.bar(x,      f1s,  w, label="F1 Score", color=PALETTE[1])
ax.bar(x + w,  aucs, w, label="AUC-ROC",  color=PALETTE[2])
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0.5, 1.05); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Model Comparison", fontsize=13, fontweight="bold", pad=12)
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"), dpi=150)
plt.close()
print("Saved: model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. BEST MODEL + HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 8: Hyperparameter Tuning (GridSearchCV)")
print("="*60)

best_name = max(results, key=lambda n: results[n]["AUC-ROC"])
print(f"Best model: {best_name}")

# Tune Random Forest (or best model)
param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    base_clf, param_grid, cv=cv,
    scoring="roc_auc", n_jobs=-1, verbose=0)
grid_search.fit(X_train_s, y_train)
print(f"Best params : {grid_search.best_params_}")
print(f"Best AUC-ROC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ─────────────────────────────────────────────────────────────────────────────
# 9. CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 9: Cross Validation")
print("="*60)

cv_scores = cross_val_score(best_model, X_train_s, y_train,
                             cv=cv, scoring="roc_auc")
print(f"CV AUC-ROC scores : {cv_scores.round(4)}")
print(f"Mean ± Std        : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 10: Evaluation Plots")
print("="*60)

y_pred  = best_model.predict(X_test_s)
y_proba = best_model.predict_proba(X_test_s)[:, 1]

# 10a — Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm  = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churn"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# 10b — ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val     = roc_auc_score(y_test, y_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color=PALETTE[0], lw=2.5, label=f"AUC = {auc_val:.4f}")
ax.plot([0,1],[0,1], "--", color=PALETTE[3], lw=1.5, label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE[0])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right"); ax.grid()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("Saved: roc_curve.png")

# 10c — Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec, prec, color=PALETTE[1], lw=2.5)
ax.fill_between(rec, prec, alpha=0.08, color=PALETTE[1])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
ax.grid()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "precision_recall_curve.png"), dpi=150)
plt.close()
print("Saved: precision_recall_curve.png")

# 10d — Feature Importance
feat_imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 7))
colors  = [PALETTE[0] if v > feat_imp.median() else PALETTE[2] for v in feat_imp]
feat_imp.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importance", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Importance Score"); ax.grid(axis="x")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("Saved: feature_importance.png")

# 10e — Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn","Churn"]))

# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE MODEL & SCALER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 11: Save model.pkl & scaler.pkl")
print("="*60)

joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler,     SCALER_PATH)
print(f"Saved model  -> {MODEL_PATH}")
print(f"Saved scaler -> {SCALER_PATH}")
print("\nTraining complete!")
