# Experiment 3: Gradient Boosting with Probability Calibration

## Overview
**Objective**: Improve model probability estimates and maintain threshold optimization gains through probability calibration

**Key Innovation**: Add CalibratedClassifierCV with sigmoid method to improve probability reliability for threshold optimization

## Rationale
Building on Iteration 2's successful threshold optimization (ROC-AUC: 0.932, Recall: 74.7%), this iteration focuses on improving the reliability of probability estimates. Exploration experiments showed that sigmoid calibration provides a +0.0007 ROC-AUC improvement while maintaining the model's discriminative power. Better calibrated probabilities will enhance threshold optimization effectiveness and provide more reliable business decision-making capabilities.

## Experimental Design

### Data Preprocessing
1. **Categorical Encoding**
   - Apply `LabelEncoder` to columns: `V2, V3, V4, V5, V7, V8, V9, V11, V16`
   - Convert string categorical values to numerical representation

2. **Special Value Handling** 
   - Create `V14_is_missing` binary indicator where `V14 == -1.0`
   - Create `V15_is_zero` binary indicator where `V15 == 0.0`

3. **Outlier Management**
   - Clip `V6` values to 1st and 99th percentiles to handle extreme range (-8019 to 98417)

### Feature Engineering
- **Approach**: Maintain current feature set to isolate calibration impact
- **Total Features**: 18 (16 original + 2 engineered binary indicators)
- **Feature List**: `V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V14_is_missing, V15_is_zero`

### Model Architecture
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

base_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5, 
    learning_rate=0.1,
    random_state=42
)

calibrated_model = CalibratedClassifierCV(
    base_estimator=base_model,
    method='sigmoid',
    cv=3
)
```

**Validation**: 5-fold Stratified Cross-Validation with shuffle, random_state=42

### Evaluation Framework

#### Primary Metrics
- **ROC-AUC** (target: ≥ 0.932) - Maintain discriminative performance
- **Brier Score** - Assess probability calibration quality (lower is better)

#### Threshold Optimization Analysis
- Test threshold range: 0.1 to 0.9 (step size: 0.01)
- Find F1-optimal, precision-recall optimal, and business-optimal thresholds
- Compare threshold stability between calibrated and uncalibrated models

#### Calibration-Specific Analysis
- **Reliability Diagrams**: Assess calibration curve quality
- **Probability Distribution Comparison**: Before vs after calibration
- **Calibration Metrics**: Reliability, sharpness, and resolution

#### Comprehensive Performance Suite
- F1-Score, Recall, Precision at optimal threshold
- Average Precision (PR-AUC)
- Accuracy at optimal threshold  
- Business impact metrics (campaign effectiveness)

### Expected Deliverables

#### Model Artifacts
- MLflow logged CalibratedClassifierCV model with proper signatures
- Underlying GradientBoostingClassifier for comparison
- Complete preprocessing pipeline for production deployment

#### Analysis Reports  
- **Calibration Analysis**: Reliability diagrams and calibration curves
- **Threshold Optimization**: Enhanced precision-recall analysis with calibrated probabilities
- **ROC Analysis**: Comparative ROC curves (calibrated vs uncalibrated)
- **Probability Distribution**: Histogram comparison of model outputs
- **Business Impact**: Quantified improvement in subscriber identification accuracy

#### Performance Insights
- Statistical significance of calibration improvement
- Optimal threshold recommendation for business deployment
- Campaign targeting effectiveness enhancement metrics

## Success Criteria

### Primary Success
- Maintain ROC-AUC ≥ 0.932 while demonstrating improved probability calibration

### Secondary Success Indicators  
- Improve Brier Score compared to Iteration 2 (0.063)
- Better calibrated probabilities shown in reliability diagrams
- Maintain or improve recall at optimal threshold (≥ 74.7%)
- Stable or enhanced business impact metrics

## Implementation Strategy

### Single Change Philosophy
This iteration introduces **only** probability calibration to isolate its impact on:
- Probability reliability and trustworthiness
- Threshold optimization effectiveness  
- Business decision-making capability

### Technical Considerations
- CalibratedClassifierCV increases training time ~3x due to internal cross-validation
- Sigmoid calibration chosen over isotonic based on exploration results
- Model provides enhanced probability estimates for real-time campaign decisions

### Production Readiness
The calibrated model will deliver:
- More reliable probability estimates for threshold-based decisions
- Better business interpretability of prediction confidence
- Enhanced campaign targeting precision through improved probability reliability

## Business Impact
This experiment directly supports the banking institution's marketing campaign optimization by providing more trustworthy probability estimates, enabling better threshold-based decisions for customer targeting and resource allocation.