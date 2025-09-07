#!/usr/bin/env python3
"""
Exploration of advanced ensemble approaches with calibration
for bank marketing dataset - Experiment 4 planning.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
    print("XGBoost available")
except ImportError:
    HAS_XGB = False
    print("XGBoost not available")

try:
    import lightgbm as lgb  
    HAS_LGB = True
    print("LightGBM available")
except ImportError:
    HAS_LGB = False
    print("LightGBM not available")

# Load the data
print("\nLoading training data...")
train_df = pd.read_csv('/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/data/train.csv')

# Basic preprocessing
def preprocess_data(df):
    """Basic preprocessing similar to previous experiments"""
    df_processed = df.copy()
    
    # Label encode categorical variables
    categorical_columns = ['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    return df_processed

# Create minimal feature engineering (best from previous test)
def create_basic_features(df):
    """Add the 2 engineered features from previous iterations"""
    df_eng = df.copy()
    df_eng['balance_positive'] = (df_eng['V6'] > 0).astype(int)
    df_eng['duration_high'] = (df_eng['V12'] > df_eng['V12'].quantile(0.75)).astype(int)
    return df_eng

train_processed = preprocess_data(train_df)
train_processed = create_basic_features(train_processed)
X = train_processed.drop(['target'], axis=1)
y = train_processed['target']

print(f"Final feature set shape: {X.shape}")

# EXPERIMENT 1: Baseline from previous iteration (GB + Calibration)
print("\n" + "="*60)
print("EXPERIMENT 1: Baseline - Calibrated Gradient Boosting")
print("="*60)

gb_base = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

gb_calibrated = CalibratedClassifierCV(gb_base, cv=3, method='sigmoid')
baseline_scores = cross_val_score(gb_calibrated, X, y, cv=3, scoring='roc_auc')
print(f"Calibrated GB ROC-AUC: {baseline_scores.mean():.4f} (+/- {baseline_scores.std()*2:.4f})")

# EXPERIMENT 2: Multiple Algorithm Ensemble with Calibration
print("\n" + "="*60)
print("EXPERIMENT 2: Multi-Algorithm Calibrated Ensemble")  
print("="*60)

algorithms = []
algorithm_names = []

# Always include Gradient Boosting
gb_cal = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    cv=3, method='sigmoid'
)
algorithms.append(('gb', gb_cal))
algorithm_names.append('Gradient Boosting')

# Add XGBoost if available
if HAS_XGB:
    xgb_cal = CalibratedClassifierCV(
        xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
        cv=3, method='sigmoid'
    )
    algorithms.append(('xgb', xgb_cal))
    algorithm_names.append('XGBoost')

# Add LightGBM if available  
if HAS_LGB:
    lgb_cal = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1),
        cv=3, method='sigmoid' 
    )
    algorithms.append(('lgb', lgb_cal))
    algorithm_names.append('LightGBM')

print(f"Available algorithms: {algorithm_names}")

if len(algorithms) > 1:
    # Create voting ensemble
    voting_ensemble = VotingClassifier(
        estimators=algorithms,
        voting='soft'  # Use probabilities for voting
    )
    
    ensemble_scores = cross_val_score(voting_ensemble, X, y, cv=3, scoring='roc_auc')
    print(f"Multi-Algorithm Ensemble ROC-AUC: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std()*2:.4f})")
    print(f"Improvement over baseline: {ensemble_scores.mean() - baseline_scores.mean():.4f}")
else:
    print("Only Gradient Boosting available - skipping ensemble")
    ensemble_scores = baseline_scores

# EXPERIMENT 3: Diversified Parameter Ensemble 
print("\n" + "="*60)
print("EXPERIMENT 3: Diversified Parameter GB Ensemble")
print("="*60)

# Create multiple GB models with different parameters
gb_models = [
    ('gb_shallow', CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42),
        cv=3, method='sigmoid')),
    ('gb_deep', CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=42),
        cv=3, method='sigmoid')),
    ('gb_fast', CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=80, learning_rate=0.15, max_depth=6, random_state=42),
        cv=3, method='sigmoid'))
]

diversified_ensemble = VotingClassifier(
    estimators=gb_models,
    voting='soft'
)

diversified_scores = cross_val_score(diversified_ensemble, X, y, cv=3, scoring='roc_auc')
print(f"Diversified GB Ensemble ROC-AUC: {diversified_scores.mean():.4f} (+/- {diversified_scores.std()*2:.4f})")
print(f"Improvement over baseline: {diversified_scores.mean() - baseline_scores.mean():.4f}")

# EXPERIMENT 4: Feature Subset Ensemble (Bagging-like approach)
print("\n" + "="*60)  
print("EXPERIMENT 4: Feature Subset Ensemble")
print("="*60)

from sklearn.ensemble import BaggingClassifier

# Create bagging ensemble with feature subsampling
bagging_cal = BaggingClassifier(
    base_estimator=CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=6, random_state=42),
        cv=2, method='sigmoid'  # Reduce CV for inner calibration
    ),
    n_estimators=3,  # Number of base models
    max_features=0.8,  # Use 80% of features per model
    random_state=42
)

bagging_scores = cross_val_score(bagging_cal, X, y, cv=3, scoring='roc_auc')
print(f"Feature Subset Ensemble ROC-AUC: {bagging_scores.mean():.4f} (+/- {bagging_scores.std()*2:.4f})")
print(f"Improvement over baseline: {bagging_scores.mean() - baseline_scores.mean():.4f}")

# Summary
print("\n" + "="*60)
print("ENSEMBLE EXPLORATION SUMMARY") 
print("="*60)
print(f"Calibrated GB (baseline):         {baseline_scores.mean():.4f}")

if len(algorithms) > 1:
    print(f"Multi-Algorithm Ensemble:         {ensemble_scores.mean():.4f} (+{ensemble_scores.mean() - baseline_scores.mean():.4f})")

print(f"Diversified Parameter Ensemble:   {diversified_scores.mean():.4f} (+{diversified_scores.mean() - baseline_scores.mean():.4f})")
print(f"Feature Subset Ensemble:          {bagging_scores.mean():.4f} (+{bagging_scores.mean() - baseline_scores.mean():.4f})")

# Identify best approach
approaches = {
    'baseline': baseline_scores.mean(),
    'diversified': diversified_scores.mean(),
    'bagging': bagging_scores.mean()
}

if len(algorithms) > 1:
    approaches['multi_algorithm'] = ensemble_scores.mean()

best_approach = max(approaches.items(), key=lambda x: x[1])
best_improvement = best_approach[1] - baseline_scores.mean()

print(f"\nBest ensemble approach: {best_approach[0]} with score {best_approach[1]:.4f}")
print(f"Improvement over baseline: {best_improvement:.4f}")
print(f"Potential boost from previous best of ~0.934 to ~{0.934 + best_improvement:.4f}")