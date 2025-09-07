# Experiment 2: Gradient Boosting with Threshold Optimization for Class Imbalance

## Experiment Overview

**Iteration**: 2  
**Primary Change**: Implement precision-recall threshold optimization to address class imbalance  
**Target Improvement**: Increase positive class recall from 47.1% to 70%+ while maintaining ROC-AUC ≥0.93

## Context from Previous Experiment

Experiment 1 achieved excellent discriminative performance (ROC-AUC = 0.935) but suffered from the classic precision-recall tradeoff in imbalanced classification:
- ✅ **Strengths**: Excellent overall discrimination, well-calibrated probabilities  
- ❌ **Limitation**: Low recall for positive class (47.1%) - missing ~53% of potential subscribers
- 🎯 **Opportunity**: Threshold optimization can improve recall significantly without architectural changes

## Experimental Evidence

Based on lightweight experiments conducted:

| Approach | ROC-AUC | Recall | Precision | F1-Score | Notes |
|----------|---------|--------|-----------|----------|--------|
| Baseline (0.5 threshold) | 0.9301 | 53.8% | 76.7% | 63.2% | Previous experiment equivalent |
| Optimal F1 Threshold | 0.9301 | 73.1% | 64.0% | 68.2% | **Best balanced performance** |
| High Recall Threshold (0.2) | 0.9301 | 84.4% | 52.9% | 65.1% | Too aggressive |
| Class Weighting | 0.9529 | 92.7% | 45.9% | 61.4% | Overly aggressive |
| SMOTE Oversampling | 0.9369 | 70.3% | 58.2% | 63.7% | Good but slower training |

**Selected Approach**: Threshold optimization provides the best balance - significant recall improvement with minimal complexity increase.

## Detailed Implementation Plan

### 1. Data Preprocessing Steps

```python
# Categorical Encoding
categorical_cols = ['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16']
for col in categorical_cols:
    X_processed[col] = LabelEncoder().fit_transform(X_processed[col].astype(str))

# Special Value Handling  
X_processed['V14_is_missing'] = (X_processed['V14'] == -1.0).astype(int)
X_processed['V15_is_zero'] = (X_processed['V15'] == 0.0).astype(int)

# Outlier Treatment
X_processed['V6'] = np.clip(X_processed['V6'], -1000, 10000)
```

### 2. Feature Engineering

- **Total Features**: 18 (16 original + 2 engineered)
- **V14_is_missing**: Binary indicator for V14 == -1.0 
- **V15_is_zero**: Binary indicator for V15 == 0.0
- **Feature Selection**: Use all features (no dimensionality reduction)

### 3. Model Configuration

```python
GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=5, 
    n_estimators=200,
    random_state=42
)
```

**Validation**: 5-fold stratified cross-validation with ROC-AUC scoring

### 4. Threshold Optimization Pipeline

```python
# 1. Generate predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 2. Create precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba, pos_label=2)

# 3. Find F1-optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# 4. Evaluate multiple thresholds
thresholds_to_test = [0.5, optimal_threshold, 0.3, 0.2]
```

### 5. Evaluation Strategy

**Primary Metric**: ROC-AUC (maintain ≥0.93)  
**Secondary Metrics**: Average Precision, F1-Score, Precision, Recall, Brier Score

**Diagnostic Analyses** (7 comprehensive plots):

1. **Threshold Performance Analysis** - Performance across multiple thresholds
2. **Enhanced Precision-Recall Curve** - With optimal threshold highlighted  
3. **Confusion Matrix Comparison** - Side-by-side at different thresholds
4. **Feature Importance Validation** - Confirm V12 remains top predictor
5. **Probability Distribution Analysis** - By true class for threshold selection
6. **Business Impact Analysis** - Campaign metrics at different thresholds
7. **Model Calibration Assessment** - Reliability of probability predictions

## Success Criteria

### Primary Success
- **Recall for positive class**: ≥70% (vs 47.1% previously)
- **ROC-AUC maintenance**: ≥0.93
- **F1-Score improvement**: ≥0.65 (vs 0.55 previously)

### Business Impact
- **Subscriber Detection**: Identify 70%+ of potential subscribers vs 47% previously
- **Campaign Efficiency**: Improved hit rate with acceptable precision tradeoff
- **Model Usability**: Configurable threshold for different business scenarios

## Expected Performance Profile

| Threshold | Use Case | Recall | Precision | F1-Score |
|-----------|----------|---------|-----------|----------|
| 0.5 | Conservative | ~54% | ~77% | ~63% |
| F1-Optimal | Balanced | ~73% | ~64% | ~68% |
| 0.3 | Aggressive | ~78% | ~58% | ~67% |
| 0.2 | High Coverage | ~84% | ~53% | ~65% |

## Implementation Notes

- **Single Change Focus**: Only threshold optimization - no other model modifications
- **Code Extension**: Build upon Experiment 1 codebase with threshold pipeline
- **MLflow Integration**: Log threshold performance and optimal threshold as metadata
- **Production Ready**: Include threshold as configurable deployment parameter

## Risk Mitigation

- **ROC-AUC Monitoring**: Ensure no degradation in discriminative performance
- **Business Alignment**: Validate precision-recall tradeoff with stakeholders  
- **Threshold Validation**: Test multiple thresholds to avoid overfitting to F1 metric
- **Calibration Check**: Ensure probability quality remains high for threshold reliability

This experiment focuses on solving the key limitation from Experiment 1 while maintaining its strengths, providing a practical improvement that can be immediately deployed to improve marketing campaign effectiveness.