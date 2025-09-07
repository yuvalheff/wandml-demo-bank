# Experiment 1: Gradient Boosting Baseline with Optimized Preprocessing

## Experiment Overview
**Objective**: Establish a high-performance baseline model for bank marketing campaign prediction using Gradient Boosting with optimized preprocessing based on comprehensive EDA and exploration experiments.

**Target Performance**: ROC-AUC ≥ 0.935 (based on exploration experiments achieving 0.9355)

## Data Preprocessing Steps

### 1. Special Value Handling
- **V14 missing value indicator**: Create binary feature `V14_is_missing = (df['V14'] == -1.0).astype(int)`
- **V15 zero value indicator**: Create binary feature `V15_is_zero = (df['V15'] == 0.0).astype(int)`
- **Rationale**: EDA revealed V14 has many -1.0 values (likely missing) and V15 has many zeros, creating indicators preserves this information

### 2. Outlier Treatment
- **V6 outlier capping**: Apply percentile-based clipping
  ```python
  v6_q99 = df['V6'].quantile(0.99)
  v6_q01 = df['V6'].quantile(0.01)
  df['V6'] = np.clip(df['V6'], v6_q01, v6_q99)
  ```
- **Rationale**: EDA showed V6 has extreme range (-8019 to 98417) with significant outliers affecting model performance

### 3. Categorical Feature Encoding
- **Method**: One-hot encoding using `pd.get_dummies(df, columns=['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16'], drop_first=True)`
- **Features to encode**: All categorical columns identified in EDA
- **Expected output**: ~44 total features after encoding

### 4. Column Alignment
- Ensure identical feature columns between train/test sets using set intersection
- Handle any encoding differences between splits

## Feature Engineering Strategy
- **Approach**: Minimal feature engineering based on exploration experiments
- **Rationale**: Complex feature engineering showed minimal improvement (+0.0004 AUC)
- **Focus**: Preserve strong predictive power of existing features, especially V12
- **No scaling needed**: Tree-based models handle mixed feature scales naturally

## Model Selection and Training

### Primary Algorithm: Gradient Boosting Classifier
**Selection rationale**: Achieved highest performance (0.9355 ROC-AUC) in exploration experiments

### Hyperparameter Tuning Grid
```python
param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.1, 0.15, 0.2], 
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
```

### Cross-Validation Strategy
- **Method**: 5-fold stratified cross-validation
- **Rationale**: Maintains class balance (88.3% vs 11.7%) across folds
- **Scoring**: ROC-AUC (primary metric)

### Baseline Comparison
- Include RandomForestClassifier with `class_weight='balanced'` as benchmark
- Expected performance: ~0.934 ROC-AUC based on exploration

## Evaluation Strategy

### Primary Evaluation Metric
- **ROC-AUC**: Optimal for imbalanced binary classification with cost-sensitive decision making

### Comprehensive Diagnostic Analysis

#### 1. Feature Importance Analysis
- **Purpose**: Validate V12 as top predictor (expected ~35% importance)
- **Output**: Ranked feature importance plot and table
- **Insight**: Confirm EDA findings and understand model decision patterns

#### 2. Performance Segmentation by Job Type (V2)
- **Analysis**: Calculate ROC-AUC for each job category with sufficient samples (n>50)
- **Expected patterns**: Students and retirees should show highest model performance
- **Purpose**: Identify customer segments where model excels or struggles

#### 3. ROC Curve and Threshold Analysis  
- **Generate**: Full ROC curve with optimal threshold identification
- **Method**: Maximize Youden's Index (TPR - FPR)
- **Business application**: Provide actionable probability thresholds

#### 4. Precision-Recall Analysis
- **Rationale**: Critical for imbalanced dataset evaluation
- **Focus**: Model performance on minority class (subscription=yes)
- **Output**: PR curve and optimal precision-recall trade-off point

#### 5. Model Calibration Assessment
- **Method**: Calibration plot and Brier score
- **Purpose**: Evaluate probability reliability for business decision-making
- **Threshold**: Well-calibrated probabilities essential for campaign targeting

#### 6. Error Analysis
- **False Positive analysis**: Identify characteristics of incorrectly predicted subscribers
- **False Negative analysis**: Understand missed subscription opportunities  
- **Pattern identification**: Common features in misclassified samples

#### 7. Learning Curves
- **Training size vs performance**: Assess if more data would improve performance
- **Overfitting detection**: Monitor train vs validation performance gaps
- **Data efficiency**: Understand minimum training data requirements

## Expected Deliverables

### 1. Model Artifact
- Optimally tuned GradientBoostingClassifier
- Performance target: ROC-AUC ≥ 0.935

### 2. Performance Report
- Cross-validation results with error bars
- Test set performance across all metrics
- Feature importance ranking with business interpretation

### 3. Diagnostic Analysis
- Comprehensive evaluation across all specified analyses
- Actionable business insights for campaign optimization
- Model limitation identification and recommendations

### 4. Business Insights
- Customer segment prioritization based on model performance
- Optimal targeting strategy recommendations
- Feature importance translated to business actionability

## Success Criteria
1. **Performance**: Achieve ROC-AUC ≥ 0.935 on test set
2. **Stability**: CV standard deviation < 0.01 indicating robust performance  
3. **Interpretability**: Clear feature importance ranking matching EDA insights
4. **Business value**: Actionable insights for marketing campaign optimization
5. **Model reliability**: Well-calibrated probability estimates for decision-making

## Risk Mitigation
- **Overfitting**: Monitor learning curves and use early stopping if needed
- **Feature drift**: Validate feature importance consistency with EDA
- **Class imbalance**: Confirm model handles minority class effectively through PR analysis
- **Generalization**: Ensure diagnostic analyses reveal no concerning performance patterns