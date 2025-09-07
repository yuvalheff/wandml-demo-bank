# Experiment 3: Gradient Boosting with Probability Calibration

## Experiment Overview

**Objective**: Improve model probability estimates and maintain threshold optimization gains through probability calibration

**Key Change**: Added CalibratedClassifierCV with sigmoid method to improve probability reliability for threshold optimization

**Hypothesis**: Sigmoid calibration would improve probability reliability without sacrificing discriminative performance, leading to better threshold optimization effectiveness.

## Results Summary

### Primary Metric Performance
- **ROC-AUC**: 0.9338 ✅ (Target: >0.932)
- **Improvement over Iteration 2**: +0.0016 (0.9322 → 0.9338)
- **Brier Score**: 0.0623 ✅ (Improved from 0.0635 in Iteration 2)

### Threshold Optimization Results
- **Optimal Threshold**: 0.245
- **F1-Score at Optimal**: 0.634 ✅ (Improved from 0.627 in Iteration 2)
- **Recall at Optimal**: 75.0% ✅ (Target: ≥74.7%, Previous: 74.7%)
- **Precision at Optimal**: 54.8% (Slight improvement from 54.0%)

### Classification Performance
- **Accuracy**: 90.9% (up from 90.7%)
- **Average Precision**: 0.733 (maintained)
- **Positive Class Recall**: 44.8% (default threshold)
- **Positive Class Precision**: 66.4% (default threshold)

## Key Findings

### 1. Successful Calibration Impact
The experiment successfully achieved its primary goal of improving probability calibration:
- **Brier Score improved** from 0.0635 to 0.0623 (lower is better)
- **ROC-AUC increased** by +0.0016, exceeding the target threshold
- Calibration provides more reliable probability estimates for business decision-making

### 2. Enhanced Threshold Optimization
Calibration improved the effectiveness of threshold optimization:
- **F1-score at optimal threshold** improved from 0.627 to 0.634 (+0.007)
- **Recall maintained** at the target level of ~75%
- Better balance between precision and recall at the optimal operating point

### 3. Computational Trade-offs
- Training time increased ~3x due to internal cross-validation in CalibratedClassifierCV
- Maintained model complexity with 18 features (16 original + 2 engineered indicators)
- Production-ready model with proper MLflow integration

## Weaknesses and Limitations

### 1. Class Imbalance Persistence
- **Severe class imbalance** (11.7% positive class) continues to challenge performance
- **Low default threshold recall** (44.8%) requires threshold optimization for practical use
- Precision-recall trade-off still requires business context for optimal threshold selection

### 2. Limited Feature Engineering
- Feature set remained static at 18 features to isolate calibration impact
- Potential for additional feature interactions or domain-specific engineering unexplored
- Missing feature importance analysis in this iteration

### 3. Calibration Method Limitation
- Only sigmoid calibration tested; isotonic calibration might offer additional benefits
- No comparison with ensemble calibration methods
- Calibration effectiveness on different data distributions not validated

## Context Notes for Continuity

### Model Architecture
- **Base Model**: GradientBoostingClassifier (200 estimators, max_depth=5, learning_rate=0.1)
- **Calibration**: CalibratedClassifierCV with sigmoid method (3-fold CV)
- **Preprocessing**: LabelEncoder for 9 categorical features, quantile clipping for V6, binary indicators for V14/V15

### Business Impact
- **Improved targeting**: Better calibrated probabilities enable more accurate campaign targeting
- **Threshold flexibility**: Optimized threshold (0.245) provides 75% recall with 54.8% precision
- **Production readiness**: MLflow model registered with proper signatures for deployment

### Technical Execution
- Experiment required 3 execution attempts due to MLflow path issues (resolved)
- All success criteria met: ROC-AUC ≥0.932, improved Brier Score, maintained recall ≥74.7%
- Comprehensive evaluation including calibration curves, threshold analysis, and prediction distributions

## Future Suggestions

### Priority 1: Advanced Feature Engineering
**Rationale**: Current feature set (18 features) may be limiting model potential
**Approach**: 
- Create interaction features between demographic and campaign variables
- Engineer temporal features from contact timing (V11, V10)
- Develop domain-specific ratios (e.g., balance-to-age, campaign intensity metrics)
- Target 25-30 features with systematic feature selection

### Priority 2: Ensemble with Calibration Variants
**Rationale**: Single calibration method may not capture all probability distribution nuances  
**Approach**:
- Ensemble of sigmoid and isotonic calibrated models
- Stack with different base algorithms (XGBoost, LightGBM) each with calibration
- Test ensemble calibration methods (e.g., temperature scaling)

### Priority 3: Advanced Class Imbalance Techniques
**Rationale**: 11.7% positive class rate creates fundamental modeling challenges
**Approach**:
- Cost-sensitive learning with business-informed cost matrix
- Advanced sampling techniques (ADASYN, BorderlineSMOTE)
- Focal loss implementation for direct imbalance handling

The calibration approach successfully improved probability reliability and threshold optimization effectiveness while maintaining strong discriminative performance. The next iteration should focus on expanding the feature space to capture more complex patterns in the banking marketing domain.