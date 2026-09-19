import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
)


def add_groups(frame):
    out = pd.DataFrame(index=frame.index)
    out['age_group'] = pd.cut(
        frame['age'], bins=[0, 40, 60, 75, 200],
        labels=['<40', '40-59', '60-74', '75+'], right=False,
    )
    out['bmi_group'] = pd.cut(
        frame['bmi'], bins=[0, 18.5, 25, 30, 40, 200],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severe obesity'],
        right=False,
    )
    out['diabetes_group'] = frame['has_diabetes'].map({0: 'No diabetes', 1: 'Diabetes'})
    return out


def group_rates(d, group_col, pred_col='y_pred'):
    from sklearn.metrics import confusion_matrix

    rows = []
    for g, sub in d.groupby(group_col, observed=True):
        tn, fp, fn, tp = confusion_matrix(
            sub['y_true'], sub[pred_col], labels=[0, 1]
        ).ravel()
        rows.append({
            'group': str(g),
            'n': len(sub),
            'base_rate': sub['y_true'].mean(),
            'flag_rate': sub[pred_col].mean(),
            'TPR': tp / (tp + fn) if tp + fn else np.nan,
            'FPR': fp / (fp + tn) if fp + tn else np.nan,
            'PPV': tp / (tp + fp) if tp + fp else np.nan,
            'accuracy': (tp + tn) / len(sub),
        })
    return pd.DataFrame(rows).set_index('group')


def fairness_summary(rates):
    tpr_gap = rates['TPR'].max() - rates['TPR'].min()
    fpr_gap = rates['FPR'].max() - rates['FPR'].min()
    return {
        'demographic_parity_diff': rates['flag_rate'].max() - rates['flag_rate'].min(),
        'disparate_impact_ratio': rates['flag_rate'].min() / rates['flag_rate'].max(),
        'equal_opportunity_diff': tpr_gap,
        'fpr_diff': fpr_gap,
        'equalised_odds_diff': max(tpr_gap, fpr_gap),
    }


def run_fairness_mitigation(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled,
    best_model,
    audit_df,
    output_csv='fairness_mitigation.csv',
):
    FEATURES = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else list(range(X_train.shape[1]))
    Xtr = X_train_scaled
    Xte = X_test_scaled
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train, columns=FEATURES)
    else:
        X_train_df = X_train

    age_train = add_groups(X_train_df)['age_group'].astype('category')
    age_test = audit_df['age_group'].astype('category')

    p_group = age_train.value_counts(normalize=True)
    p_y = y_train.value_counts(normalize=True)
    p_joint = pd.crosstab(age_train, y_train, normalize=True)

    weights = np.array([
        p_group[g] * p_y[yv] / p_joint.loc[g, yv]
        for g, yv in zip(age_train, y_train)
    ])

    model_reweighted = clone(best_model)
    model_reweighted.fit(Xtr, y_train, sample_weight=weights)
    prob_rw = model_reweighted.predict_proba(Xte)[:, 1]

    oof_prob = cross_val_predict(
        clone(best_model), Xtr, y_train, cv=cv, method='predict_proba'
    )[:, 1]

    target_tpr = recall_score(y_train, (oof_prob >= 0.5).astype(int))

    group_thresholds = {}
    for g in age_train.cat.categories:
        m = (age_train == g).values
        if m.sum() == 0:
            continue
        candidates = np.arange(0.05, 0.96, 0.01)
        tprs = np.array([
            recall_score(y_train[m], (oof_prob[m] >= t).astype(int))
            for t in candidates
        ])
        group_thresholds[g] = float(candidates[np.argmin(np.abs(tprs - target_tpr))])

    thr_test = (
        age_test.astype(str)
                .map({str(k): v for k, v in group_thresholds.items()})
                .values
    )
    pred_thr = (audit_df['y_prob'].values >= thr_test).astype(int)

    def evaluate(pred, prob, label):
        d = audit_df[['age_group', 'y_true']].copy()
        d['pred'] = pred
        rates = group_rates(d, 'age_group', 'pred')
        fs = fairness_summary(rates)
        return {
            'approach': label,
            'AUC': roc_auc_score(y_test, prob),
            'accuracy': accuracy_score(y_test, pred),
            'recall': recall_score(y_test, pred),
            'precision': precision_score(y_test, pred),
            **fs,
        }

    mitigation = pd.DataFrame([
        evaluate(audit_df['y_pred'].values, audit_df['y_prob'].values,
                 'Baseline (threshold 0.5)'),
        evaluate((prob_rw >= 0.5).astype(int), prob_rw,
                 'A: Group reweighting'),
        evaluate(pred_thr, audit_df['y_prob'].values,
                 'B: Age-group thresholds'),
    ]).set_index('approach')

    mitigation.to_csv(output_csv)
    print(mitigation.round(3))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    mitigation[['equal_opportunity_diff', 'fpr_diff', 'equalised_odds_diff']].plot.bar(
        ax=axes[0], rot=15
    )
    axes[0].axhline(0.10, ls='--', color='grey')
    axes[0].set_title('Fairness gaps across age groups (lower = fairer)')

    mitigation[['AUC', 'accuracy', 'recall']].plot.bar(ax=axes[1], rot=15)
    axes[1].set_ylim(0.5, 1)
    axes[1].set_title('Performance cost of each mitigation')

    for ax in axes:
        ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.show()

    return mitigation


if __name__ == '__main__':
    print("Import this module from the notebook to run Section 5.3 mitigation.")
