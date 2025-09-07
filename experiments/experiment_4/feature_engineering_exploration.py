#!/usr/bin/env python3
"""
Lightweight exploration of advanced feature engineering approaches
for bank marketing dataset - Experiment 4 planning.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading training data...")
train_df = pd.read_csv('/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/data/train.csv')
print(f"Training data shape: {train_df.shape}")
print("\nFirst few rows:")
print(train_df.head(3))

# Basic preprocessing function
def preprocess_data(df):
    """Basic preprocessing similar to previous experiments"""
    df_processed = df.copy()
    
    # Label encode categorical variables
    categorical_columns = ['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    return df_processed

# Preprocess basic data
train_processed = preprocess_data(train_df)
X_basic = train_processed.drop(['target'], axis=1)
y = train_processed['target']

print(f"\nTarget distribution:")
print(y.value_counts(normalize=True))

# Baseline model with basic features (similar to previous experiments)
print("\n" + "="*60)
print("BASELINE: Basic features (16 original)")
print("="*60)

gb_baseline = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

baseline_scores = cross_val_score(gb_baseline, X_basic, y, cv=3, scoring='roc_auc')
print(f"Baseline ROC-AUC: {baseline_scores.mean():.4f} (+/- {baseline_scores.std()*2:.4f})")

# EXPERIMENT 1: Interaction Features
print("\n" + "="*60)
print("EXPERIMENT 1: Demographic-Campaign Interactions")
print("="*60)

def create_interaction_features(df):
    """Create interaction features between key variables"""
    df_int = df.copy()
    
    # Age groups for interactions
    df_int['age_group'] = pd.cut(df_int['V1'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4])
    df_int['age_group'] = df_int['age_group'].astype(int)
    
    # Job-Age interactions (students vs others)
    df_int['job_age_interaction'] = df_int['V2'] * df_int['age_group']
    
    # Campaign intensity (V12) interactions with demographics
    df_int['duration_job_interaction'] = df_int['V12'] * df_int['V2']
    df_int['duration_education_interaction'] = df_int['V12'] * df_int['V4']
    
    # Financial status interactions
    df_int['balance_loan_interaction'] = df_int['V6'] * df_int['V7']
    
    return df_int

X_interactions = create_interaction_features(X_basic)
print(f"Features after interactions: {X_interactions.shape[1]}")

interaction_scores = cross_val_score(gb_baseline, X_interactions, y, cv=3, scoring='roc_auc')
print(f"With Interactions ROC-AUC: {interaction_scores.mean():.4f} (+/- {interaction_scores.std()*2:.4f})")
print(f"Improvement: {interaction_scores.mean() - baseline_scores.mean():.4f}")

# EXPERIMENT 2: Temporal Features  
print("\n" + "="*60)
print("EXPERIMENT 2: Temporal Feature Engineering")
print("="*60)

def create_temporal_features(df):
    """Create temporal features from contact timing"""
    df_temp = df.copy()
    
    # Day of month patterns
    df_temp['is_month_start'] = (df_temp['V10'] <= 10).astype(int)
    df_temp['is_month_middle'] = ((df_temp['V10'] > 10) & (df_temp['V10'] <= 20)).astype(int) 
    df_temp['is_month_end'] = (df_temp['V10'] > 20).astype(int)
    
    # Month seasonality (from V11)
    df_temp['is_spring'] = df_temp['V11'].isin([2, 3, 4]).astype(int)  # Mar, Apr, May
    df_temp['is_summer'] = df_temp['V11'].isin([5, 6, 7]).astype(int)  # Jun, Jul, Aug  
    df_temp['is_fall'] = df_temp['V11'].isin([8, 9, 10]).astype(int)   # Sep, Oct, Nov
    df_temp['is_winter'] = df_temp['V11'].isin([11, 0, 1]).astype(int) # Dec, Jan, Feb
    
    return df_temp

X_temporal = create_temporal_features(X_basic)
print(f"Features after temporal engineering: {X_temporal.shape[1]}")

temporal_scores = cross_val_score(gb_baseline, X_temporal, y, cv=3, scoring='roc_auc')
print(f"With Temporal ROC-AUC: {temporal_scores.mean():.4f} (+/- {temporal_scores.std()*2:.4f})")
print(f"Improvement: {temporal_scores.mean() - baseline_scores.mean():.4f}")

# EXPERIMENT 3: Domain-Specific Ratios
print("\n" + "="*60)  
print("EXPERIMENT 3: Domain-Specific Financial Ratios")
print("="*60)

def create_financial_ratios(df):
    """Create domain-specific financial ratios"""
    df_ratio = df.copy()
    
    # Campaign efficiency ratios
    df_ratio['duration_per_contact'] = df_ratio['V12'] / (df_ratio['V13'] + 1)  # Avoid division by zero
    df_ratio['success_rate_proxy'] = df_ratio['V12'] / (df_ratio['V14'].replace(-1, 0) + 1)
    
    # Financial status indicators
    df_ratio['balance_positive'] = (df_ratio['V6'] > 0).astype(int)
    df_ratio['balance_magnitude'] = np.log1p(np.abs(df_ratio['V6']))
    
    # Contact method effectiveness proxy
    df_ratio['contact_effectiveness'] = df_ratio['V9'] * df_ratio['V12']
    
    return df_ratio

X_ratios = create_financial_ratios(X_basic)
print(f"Features after financial ratios: {X_ratios.shape[1]}")

ratio_scores = cross_val_score(gb_baseline, X_ratios, y, cv=3, scoring='roc_auc')
print(f"With Financial Ratios ROC-AUC: {ratio_scores.mean():.4f} (+/- {ratio_scores.std()*2:.4f})")
print(f"Improvement: {ratio_scores.mean() - baseline_scores.mean():.4f}")

# EXPERIMENT 4: Combined Advanced Features
print("\n" + "="*60)
print("EXPERIMENT 4: Combined Advanced Feature Engineering")
print("="*60)

def create_combined_features(df):
    """Combine the most promising feature engineering approaches"""
    df_combined = df.copy()
    
    # From interactions: Most promising ones
    df_combined['age_group'] = pd.cut(df_combined['V1'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4])
    df_combined['age_group'] = df_combined['age_group'].astype(int)
    df_combined['duration_job_interaction'] = df_combined['V12'] * df_combined['V2']
    df_combined['duration_education_interaction'] = df_combined['V12'] * df_combined['V4']
    
    # From temporal: Most useful ones
    df_combined['is_month_end'] = (df_combined['V10'] > 20).astype(int)
    df_combined['is_summer'] = df_combined['V11'].isin([5, 6, 7]).astype(int)
    
    # From ratios: Most informative ones  
    df_combined['duration_per_contact'] = df_combined['V12'] / (df_combined['V13'] + 1)
    df_combined['balance_positive'] = (df_combined['V6'] > 0).astype(int)
    df_combined['balance_magnitude'] = np.log1p(np.abs(df_combined['V6']))
    
    return df_combined

X_combined = create_combined_features(X_basic)
print(f"Features after combined engineering: {X_combined.shape[1]}")

combined_scores = cross_val_score(gb_baseline, X_combined, y, cv=3, scoring='roc_auc')
print(f"Combined Features ROC-AUC: {combined_scores.mean():.4f} (+/- {combined_scores.std()*2:.4f})")
print(f"Improvement: {combined_scores.mean() - baseline_scores.mean():.4f}")

# Summary
print("\n" + "="*60)
print("FEATURE ENGINEERING EXPLORATION SUMMARY")
print("="*60)
print(f"Baseline (16 features):           {baseline_scores.mean():.4f}")
print(f"+ Interaction Features:           {interaction_scores.mean():.4f} (+{interaction_scores.mean() - baseline_scores.mean():.4f})")
print(f"+ Temporal Features:              {temporal_scores.mean():.4f} (+{temporal_scores.mean() - baseline_scores.mean():.4f})")
print(f"+ Financial Ratios:               {ratio_scores.mean():.4f} (+{ratio_scores.mean() - baseline_scores.mean():.4f})")
print(f"+ Combined Advanced Features:     {combined_scores.mean():.4f} (+{combined_scores.mean() - baseline_scores.mean():.4f})")

# Identify best approach
improvements = {
    'interactions': interaction_scores.mean() - baseline_scores.mean(),
    'temporal': temporal_scores.mean() - baseline_scores.mean(), 
    'ratios': ratio_scores.mean() - baseline_scores.mean(),
    'combined': combined_scores.mean() - baseline_scores.mean()
}

best_approach = max(improvements.items(), key=lambda x: x[1])
print(f"\nBest approach: {best_approach[0]} with improvement of {best_approach[1]:.4f}")
print(f"This represents a potential boost from previous best of ~0.934 to ~{0.934 + best_approach[1]:.4f}")