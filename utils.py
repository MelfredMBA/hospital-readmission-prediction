# src/model_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import shap
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

def train_model(data_path='data/hospital_readmission.csv'):
    """Train the readmission prediction model"""
    # Load data
    df = pd.read_csv(data_path)
    
# 4. Data Cleaning

print("Check for missing values: ")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
            print(f"  {col}: Filled missing values with median")
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"  {col}: Filled missing values with mode")

if target_col in df.columns:
    df = df.dropna(subset=[target_col])
    print(f"Removed rows with missing target. New shape: {df.shape}")

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower, upper)
    return df

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != target_col:
        df = cap_outliers(df, col)
print(f"\nOutliers capped for {len(numeric_cols)-1} numeric columns")

# 6. Check contributing features.

target_col = 'readmitted_30days'

if 'age' in df.columns:
    df['age'] = df['age'].astype(str).str.replace('_err', '').str.replace('_err', '')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age'] = df['age'].fillna(df['age'].median())

if 'length_of_stay_days' in df.columns:
    df['prolonged_stay'] = (df['length_of_stay_days'] >= 7).astype(int)

if 'prev_admissions' in df.columns:
    df['frequent_admissions'] = (df['prev_admissions'] >= 3).astype(int)

if 'num_diagnoses' in df.columns:
    df['high_diagnoses'] = (df['num_diagnoses'] >= 10).astype(int)

if 'num_medications' in df.columns:
    df['high_medications'] = (df['num_medications'] >= 15).astype(int)

if 'glucose_level' in df.columns:
    df['high_glucose'] = (df['glucose_level'] >= 200).astype(int)

if 'bmi' in df.columns:
    df['obese'] = (df['bmi'] >= 30).astype(int)

df['risk_score'] = (
    0.25 * (df['prev_admissions'] / 10) +
    0.20 * (df['length_of_stay_days'] / 30) +
    0.15 * (df['num_diagnoses'] / 15) +
    0.15 * (df['num_medications'] / 20) +
    0.15 * (df['glucose_level'] / 300) +
    0.10 * (df['bmi'] / 50)
)

feature_cols = [
    'age',
    'length_of_stay_days',
    'num_diagnoses',
    'num_medications',
    'prev_admissions',
    'glucose_level', #Uses data from diabetes.
    'bmi',
    'has_diabetes',
    'discharge_type',
    'prolonged_stay',
    'frequent_admissions',
    'high_diagnoses',
    'high_medications',
    'high_glucose',
    'obese'
]

existing_features = [col for col in feature_cols if col in df.columns]
missing_features = [col for col in feature_cols if col not in df.columns]

if missing_features:
    print(f"Warning: These features are missing: {missing_features}")
    for col in missing_features:
        df[col] = 0

print(f"\nFeature engineering complete. Selected {len(feature_cols)} features.")
print(f"Features: {feature_cols}")

X = df[feature_cols]
y = df[target_col]


# 8. Model Training. Evaluation metrics visualization.
# Class imbalance is handles by SMOTE. In the dataset, only 23% was readmitted. Through the use of this, it fixes the ration between the admission percentage.
# Uses 4 out 5 models mentioned in the assessment
# random_state = 42 is used because the use of random seed helps in reproducibility. This ensures that no matter who runs the program, it will provide consistent outputs.

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
}

results = {}
best_model = None
best_auc = 0
best_model_name = None

print("\nMODEL TRAINING AND EVALUATION")
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc
    }

    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUC-ROC: {auc:.4f}" if auc else "  AUC-ROC: N/A")

    if auc and auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name
    
    # Save models
    with open('models/readmission_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f)
    
    return best_model, scaler, feature_info

if __name__ == '__main__':
    train_model()
