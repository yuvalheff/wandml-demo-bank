import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Load data quickly
train_df = pd.read_csv('/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/data/train.csv')
test_df = pd.read_csv('/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/data/test.csv')

# Basic preprocessing
def preprocess(df):
    df = df.copy()
    df['V14_is_missing'] = (df['V14'] == -1.0).astype(int)
    df['V15_is_zero'] = (df['V15'] == 0.0).astype(int)
    v6_q99 = df['V6'].quantile(0.99)
    v6_q01 = df['V6'].quantile(0.01)
    df['V6'] = np.clip(df['V6'], v6_q01, v6_q99)
    categorical_cols = ['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16']
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

train_proc = preprocess(train_df)
test_proc = preprocess(test_df)

# Align columns
common_cols = set(train_proc.columns) & set(test_proc.columns)
train_proc = train_proc[sorted(common_cols)]
test_proc = test_proc[sorted(common_cols)]

X_train = train_proc.drop('target', axis=1)
y_train = train_proc['target']
X_test = test_proc.drop('target', axis=1)
y_test = test_proc['target']

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {1: class_weights[0], 2: class_weights[1]}

# Best models from experiments
rf_best = RandomForestClassifier(n_estimators=200, class_weight=class_weight_dict, random_state=42)
gb_best = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

print("=== FINAL RESULTS ===")
rf_best.fit(X_train, y_train)
rf_pred = rf_best.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)

gb_best.fit(X_train, y_train)  
gb_pred = gb_best.predict_proba(X_test)[:, 1]
gb_auc = roc_auc_score(y_test, gb_pred)

print(f"Random Forest AUC: {rf_auc:.4f}")
print(f"Gradient Boosting AUC: {gb_auc:.4f}")

# Top features
rf_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Features:")
print(rf_importance.head(10))
print(f"\nBest model: {'Gradient Boosting' if gb_auc > rf_auc else 'Random Forest'}")
print(f"Best AUC: {max(rf_auc, gb_auc):.4f}")