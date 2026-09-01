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
import os
import warnings
warnings.filterwarnings('ignore')

def train_model(data_path='data/hospital_readmission.csv'):
    """Train the readmission prediction model"""
    
    print("="*60)
    print("TRAINING HOSPITAL READMISSION PREDICTION MODEL")
    print("="*60)
    
    print("\n[1] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n[2] Cleaning data...")
    target_col = 'readmitted_30days'
    
    # Check for missing values
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
        print(f"  Removed rows with missing target. New shape: {df.shape}")
    
    # Cap outliers
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
    print(f"  Outliers capped for {len(numeric_cols)-1} numeric columns")
    
    print("\n[3] Feature engineering...")
    
    # Handle age column
    if 'age' in df.columns:
        df['age'] = df['age'].astype(str).str.replace('_err', '').str.replace('_err', '')
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].median())
    
    # Create derived features
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
    
    # Create risk score
    if all(col in df.columns for col in ['prev_admissions', 'length_of_stay_days', 
                                          'num_diagnoses', 'num_medications', 
                                          'glucose_level', 'bmi']):
        df['risk_score'] = (
            0.25 * (df['prev_admissions'] / 10) +
            0.20 * (df['length_of_stay_days'] / 30) +
            0.15 * (df['num_diagnoses'] / 15) +
            0.15 * (df['num_medications'] / 20) +
            0.15 * (df['glucose_level'] / 300) +
            0.10 * (df['bmi'] / 50)
        )
    
    # Define feature columns
    feature_cols = [
        'age',
        'length_of_stay_days',
        'num_diagnoses',
        'num_medications',
        'prev_admissions',
        'glucose_level',
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
    
    # Handle missing features
    existing_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"  Warning: These features are missing: {missing_features}")
        for col in missing_features:
            df[col] = 0
    
    print(f"  Feature engineering complete. Using {len(feature_cols)} features.")
    print(f"  Features: {feature_cols}")
    
    X = df[feature_cols]
    y = df[target_col]
    
    print("\n[4] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Features scaled")
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(10, len(feature_cols)))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    selected_features = np.array(feature_cols)[selector.get_support()]
    print(f"  Selected {len(selected_features)} features: {selected_features}")
    
    print("\n[5] Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"  Balanced training set: {len(X_train_balanced)} samples")
    
    print("\n[6] Training models...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, 
                                  scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
    }
    
    results = {}
    best_model = None
    best_auc = 0
    best_model_name = None
    
    print("\nMODEL TRAINING AND EVALUATION")
    for name, model in models.items():
        print(f"\n  Training {name}...")
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
        
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    F1 Score: {f1:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    AUC-ROC: {auc:.4f}" if auc else "    AUC-ROC: N/A")
        
        if auc and auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name
    
    print(f"\n  ✅ Best model: {best_model_name} (AUC: {best_auc:.4f})")
    
    print("\n[7] Saving models to 'models/' directory...")
    
    os.makedirs('models', exist_ok=True)
    
    with open('models/readmission_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("  ✅ Saved: models/readmission_model.pkl")
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  ✅ Saved: models/scaler.pkl")
    
    feature_info = {
        'features': feature_cols,
        'selected_features': selected_features.tolist(),
        'model_type': best_model_name,
        'metrics': results[best_model_name],
        'dataset_shape': df.shape,
        'readmission_rate': float(y.mean()),
        'feature_columns': feature_cols,
        'n_features': len(feature_cols)
    }
    
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print("  ✅ Saved: models/feature_info.json")
    
    print("TRAINING COMPLETE!")
    print(f"\n📊 Best Model: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.2%}")
    print(f"   AUC-ROC: {results[best_model_name]['auc_roc']:.4f}")
    print(f"   F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"\n📁 Models saved in: models/")
    print("   - readmission_model.pkl")
    print("   - scaler.pkl")
    print("   - feature_info.json")
    print("\n✅ You can now run: python src/app.py")
    
    return best_model, scaler, feature_info

if __name__ == '__main__':
    train_model()
