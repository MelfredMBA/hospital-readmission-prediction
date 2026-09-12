from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import pickle
import json

# Model Training. Evaluation metrics visualization.
# Uses 4 out 5 models mentioned in the assessment
# random_state = 42 is used because the use of random seed helps in reproducibility. This ensures that no matter who runs the program, it will provide consistent outputs.

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
    model.fit(X_train_scaled, y_train)
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

# Model saving.
# For measuring a scale of 0 (no readmission) to 1 (will be readmitted), AUC performs well as it follows a ranking order (low, moderate, and high).

with open('readmission_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

feature_info = {
    'features': feature_cols,
    'selected_features': selected_features.tolist(),
    'model_type': best_model_name,
    'metrics': results[best_model_name] if best_model_name else {},
    'dataset_shape': df.shape,
    'readmission_rate': float(y.mean()),
    'feature_columns': feature_cols,
    'n_features': len(feature_cols)
}

with open('feature_info.json', 'w') as f:
    json.dump(feature_info, f)

print(f"\nBest model: {best_model_name} (AUC: {best_auc:.4f})")
print("Model artifacts saved successfully!")
print(f"Number of features: {len(feature_cols)}")
