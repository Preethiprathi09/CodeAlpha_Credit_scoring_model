# credit_scoring.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

# -------------------------
# 1) Prepare dataset
# -------------------------
# If you have your own dataset, replace this block with pd.read_csv('yourfile.csv')
data = pd.DataFrame({
    'income': [35000, 60000, 45000, 80000, 20000, 70000, 50000, 40000, 90000, 30000,
               48000, 52000, 61000, 28000, 77000, 66000, 35000, 43000, 72000, 26000],
    'age': [25, 35, 29, 45, 22, 41, 33, 27, 50, 24, 31, 37, 42, 23, 48, 36, 26, 30, 40, 21],
    'debt': [5000, 10000, 7000, 20000, 3000, 15000, 8000, 6000, 25000, 4000,
             9000, 11000, 7000, 2000, 21000, 12000, 4500, 6500, 16000, 3500],
    'payment_history': [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    'loan_amount': [10000, 20000, 15000, 25000, 5000, 22000, 18000, 12000, 30000, 8000,
                    14000, 16000, 20000, 6000, 24000, 18000, 9000, 13000, 19000, 7000],
    # target: 1 = creditworthy, 0 = not
    'creditworthy': [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
})

# -------------------------
# 2) Features & Target
# -------------------------
X = data[['income', 'age', 'debt', 'payment_history', 'loan_amount']]
y = data['creditworthy']

# -------------------------
# 3) Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# -------------------------
# 4) Scale features
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs('artifacts', exist_ok=True)
joblib.dump(scaler, 'artifacts/scaler.pkl')

# -------------------------
# 5) Define models to train
# -------------------------
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# container for metrics
results = []

# -------------------------
# 6) Train, evaluate, save each model
# -------------------------
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # For ROC-AUC we need probability estimates; if not available, use decision function fallback
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test_scaled)
    else:
        # fallback: use predictions as probabilities (not ideal)
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc = float('nan')

    results.append({
        'model': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc
    })

    # save this model
    joblib.dump(model, f'artifacts/{name}.pkl')
    print(f"Trained and saved: {name}")

# -------------------------
# 7) Save metrics and pick best model
# -------------------------
results_df = pd.DataFrame(results).sort_values(by='roc_auc', ascending=False).reset_index(drop=True)
results_df.to_csv('artifacts/metrics.csv', index=False)

# choose best model by ROC-AUC (fallback to F1 if ROC-AUC nan or tie)
best_model_name = results_df.loc[0, 'model']
best_roc = results_df.loc[0, 'roc_auc']

# if ROC-AUC is NaN (rare), use f1
if pd.isna(best_roc):
    results_df = results_df.sort_values(by='f1_score', ascending=False).reset_index(drop=True)
    best_model_name = results_df.loc[0, 'model']

# copy the best model to model.pkl (main file used by app)
best_model = joblib.load(f'artifacts/{best_model_name}.pkl')
joblib.dump(best_model, 'artifacts/model.pkl')

print("\nModel comparison:")
print(results_df)
print(f"\nBest model selected: {best_model_name} -> saved as artifacts/model.pkl")
print("All artifacts saved inside the 'artifacts' folder (scaler, models, metrics.csv).")
