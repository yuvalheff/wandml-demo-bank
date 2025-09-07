#!/usr/bin/env python3
"""
Quick fixed version focusing on the promising multi-algorithm ensemble
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb  
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# Load and preprocess data
train_df = pd.read_csv('/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/data/train.csv')

def preprocess_data(df):
    df_processed = df.copy()
    categorical_columns = ['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16']
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    return df_processed

def create_basic_features(df):
    df_eng = df.copy()
    df_eng['balance_positive'] = (df_eng['V6'] > 0).astype(int)
    df_eng['duration_high'] = (df_eng['V12'] > df_eng['V12'].quantile(0.75)).astype(int)
    return df_eng

train_processed = preprocess_data(train_df)
train_processed = create_basic_features(train_processed)
X = train_processed.drop(['target'], axis=1)
y = train_processed['target']

print("MULTI-ALGORITHM ENSEMBLE RESULTS:")
print("="*50)

# Baseline
gb_cal = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    cv=3, method='sigmoid'
)
baseline_scores = cross_val_score(gb_cal, X, y, cv=3, scoring='roc_auc')
print(f"Calibrated GB baseline: {baseline_scores.mean():.4f}")

# Multi-algorithm ensemble
algorithms = [('gb', gb_cal)]

if HAS_XGB:
    xgb_cal = CalibratedClassifierCV(
        xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss'),
        cv=3, method='sigmoid'
    )
    algorithms.append(('xgb', xgb_cal))

if HAS_LGB:
    lgb_cal = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1),
        cv=3, method='sigmoid' 
    )
    algorithms.append(('lgb', lgb_cal))

voting_ensemble = VotingClassifier(estimators=algorithms, voting='soft')
ensemble_scores = cross_val_score(voting_ensemble, X, y, cv=3, scoring='roc_auc')

print(f"Multi-Algorithm Ensemble: {ensemble_scores.mean():.4f}")
print(f"Improvement: +{ensemble_scores.mean() - baseline_scores.mean():.4f}")
print(f"Available algorithms: {len(algorithms)} (GB, XGB, LGB)")

# Expected final performance
print(f"\nPrevious best (Exp 3): ~0.9338")
print(f"Expected with ensemble: ~{0.9338 + (ensemble_scores.mean() - baseline_scores.mean()):.4f}")