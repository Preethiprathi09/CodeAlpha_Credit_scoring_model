# CodeAlpha_Credit_scoring_model
The Credit Scoring Model is a Machine Learning project developed as part of the CodeAlpha internship, designed to predict an individual's creditworthiness based on their financial data. It helps financial institutions assess loan eligibility and risk using classification algorithms

Objective

Predict whether an individual is creditworthy (1) or not creditworthy (0) by analyzing their:

Annual income

Age

Debt

Payment history

Loan amount

⚙️ Approach

This project follows a complete end-to-end ML workflow:

Data preprocessing using pandas and StandardScaler

Model training with multiple classification algorithms:

Logistic Regression

Decision Tree

Random Forest

Model evaluation using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Model selection — the best-performing model is automatically saved

Streamlit web app for interactive predictions

🧩 Key Features

Multiple model comparison with metrics summary

Best model auto-selection (based on ROC-AUC)

Clean, user-friendly Streamlit interface

Real-time creditworthiness prediction

Reusable saved artifacts (model.pkl, scaler.pkl, and metrics.csv)

🗂️ Project Structure
CodeAlpha_CreditScoring/
│
├── credit_scoring.py     # Main training script
├── app.py                # Streamlit web application
├── artifacts/
│   ├── model.pkl         # Best trained model
│   ├── scaler.pkl        # Scaler used for feature normalization
│   ├── metrics.csv       # Model comparison results
│   ├── LogisticRegression.pkl
│   ├── DecisionTree.pkl
│   └── RandomForest.pkl
