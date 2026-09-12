# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import shap
import pickle
import json
import warnings
import random
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Data Cleaning (Nulls and Outliers)

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

# Check contributing features (Binning and Domain Features)

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

# Train-test split. It uses features with highest significance according to KBest (Feature selection and scaling)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


selector = SelectKBest(score_func=f_classif, k=min(10, len(feature_cols)))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
selected_features = np.array(feature_cols)[selector.get_support()]


print(f"\nSelected features ({len(selected_features)}): {selected_features}")
