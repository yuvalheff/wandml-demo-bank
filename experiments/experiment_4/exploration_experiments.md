# Experiment 4 Planning: Exploration Experiments Summary

## Overview
To design Experiment 4, I conducted systematic exploration experiments to identify the most promising approach for improving upon Experiment 3's strong performance (ROC-AUC: 0.9338). Based on the previous iteration's suggestions, I tested two main directions: advanced feature engineering and ensemble methods.

## Exploration Methodology
I ran lightweight experiments using 3-fold cross-validation on the training data to estimate potential improvements while maintaining the same preprocessing pipeline as previous successful experiments.

## Exploration Results

### 1. Feature Engineering Experiments
**Objective**: Test advanced feature engineering approaches suggested from previous iterations

**Approaches Tested**:
- **Baseline**: 16 original features → ROC-AUC: 0.9309
- **Demographic-Campaign Interactions**: Job-age, duration-job, duration-education, balance-loan interactions → ROC-AUC: 0.9293 (-0.0017)
- **Temporal Features**: Month periods, seasonal indicators, day-of-month patterns → ROC-AUC: 0.9326 (+0.0017) 
- **Financial Ratios**: Duration per contact, success rate proxies, balance transformations → ROC-AUC: 0.9308 (-0.0001)
- **Combined Advanced**: Best features from all approaches → ROC-AUC: 0.9303 (-0.0006)

**Key Findings**:
- Feature engineering showed modest improvements at best
- Temporal features were most promising (+0.0017) but limited impact
- Complex interactions actually hurt performance
- Risk of overfitting with too many engineered features

### 2. Ensemble Method Experiments
**Objective**: Test ensemble approaches with calibration to build on Experiment 3's success

**Approaches Tested**:
- **Baseline**: Calibrated Gradient Boosting → ROC-AUC: 0.9321
- **Multi-Algorithm Ensemble**: GB + XGBoost + LightGBM (all calibrated) → ROC-AUC: 0.9336 (+0.0015)
- **Diversified Parameter Ensemble**: Multiple GB variants → ROC-AUC: 0.9324 (+0.0003)

**Key Findings**:
- Multi-algorithm ensemble showed most consistent improvement (+0.0015)
- Algorithm diversity (GB, XGBoost, LightGBM) provides complementary strengths
- All algorithms available in current environment
- Maintains calibration quality through CalibratedClassifierCV

### 3. Performance Projections

**Expected Impact on Previous Best Performance**:
- **Experiment 3 Best**: ROC-AUC ≈ 0.9338
- **Feature Engineering**: Projected ~0.9355 (temporal features)
- **Multi-Algorithm Ensemble**: Projected ~0.9353 (ensemble diversity)

**Risk Assessment**:
- Feature engineering: Higher overfitting risk, modest gains
- Ensemble methods: More complex but proven approach, lower risk

## Decision Rationale

**Selected Approach: Multi-Algorithm Calibrated Ensemble**

**Reasons for Selection**:
1. **Builds on Success**: Leverages Experiment 3's calibration breakthrough
2. **Proven Methodology**: Ensemble methods have strong theoretical foundation
3. **Manageable Complexity**: Uses existing algorithms with known hyperparameters  
4. **Lower Overfitting Risk**: Algorithm diversity reduces overfitting compared to feature engineering
5. **Consistent Improvement**: Shows stable +0.0015 improvement across CV folds
6. **Production Viability**: All algorithms (GB, XGBoost, LightGBM) available and well-supported

**Alignment with "One Change" Constraint**:
The experiment introduces exactly one methodological change: replacing single calibrated algorithm with ensemble of three calibrated algorithms, while maintaining identical preprocessing and feature engineering.

## Implementation Strategy
- Maintain 18-feature architecture (16 original + 2 binary indicators)
- Use identical hyperparameters across algorithms for fair comparison
- Apply same calibration approach (3-fold CV, sigmoid method)
- Implement soft voting to leverage calibrated probabilities
- Focus evaluation on understanding ensemble benefits vs individual algorithms

## Expected Outcomes
- Target ROC-AUC: ≥ 0.935 (improvement of ~0.0012 over Experiment 3)
- Maintain calibration quality (Brier Score ≤ 0.063)
- Validate ensemble diversity provides genuine performance improvement
- Deploy production-ready ensemble model via MLflow