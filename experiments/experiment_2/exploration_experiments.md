# Exploration Experiments for Experiment 2 Planning

## Context and Objective

Based on Experiment 1's excellent ROC-AUC performance (0.935) but poor recall for positive class (47.1%), I conducted systematic exploration to identify the most effective approach for addressing class imbalance while maintaining discriminative performance.

## Experimental Methodology

All experiments used:
- Same preprocessing as Experiment 1 (LabelEncoder, special value handling, V6 clipping)
- Gradient Boosting with learning_rate=0.1, max_depth=5, n_estimators=100
- Evaluation on original training data for consistent comparison
- Focus on recall improvement while monitoring ROC-AUC and precision

## Experiments Conducted

### Experiment 1: Baseline (No Class Imbalance Handling)
**Configuration**: Standard Gradient Boosting with default 0.5 threshold
**Results**:
- ROC-AUC: 0.9301 ± 0.0041 (CV)
- Positive Class Metrics: Precision=76.7%, Recall=53.8%, F1=63.2%
- Analysis: Confirms the baseline performance matches Experiment 1 expectations

### Experiment 2: Class Weighting  
**Configuration**: Sample weighting with balanced class weights (1: 0.566, 2: 4.274)
**Results**:
- ROC-AUC: 0.9529 (improved discrimination)
- Positive Class Metrics: Precision=45.9%, Recall=92.7%, F1=61.4%
- Analysis: Too aggressive - high recall but very low precision makes it impractical

### Experiment 3: SMOTE Oversampling (0.5 ratio)
**Configuration**: SMOTE with 50% minority class ratio
**Results**:
- ROC-AUC: 0.9369 (slight decrease)
- Positive Class Metrics: Precision=58.2%, Recall=70.3%, F1=63.7%
- Analysis: Good balance but requires retraining and increases complexity

### Experiment 4: Threshold Optimization
**Configuration**: Baseline model with systematic threshold analysis
**Results**:
- Default (0.5): Precision=76.7%, Recall=53.8%, F1=63.2%
- Optimal F1 (0.337): Precision=64.0%, Recall=73.1%, F1=68.2%
- High Recall (0.2): Precision=52.9%, Recall=84.4%, F1=65.1%
- Analysis: **Best approach** - significant recall improvement with minimal complexity

### Experiment 5: SMOTE + Threshold Optimization
**Configuration**: Conservative SMOTE (0.3 ratio) combined with threshold optimization  
**Results**:
- ROC-AUC: 0.9423
- Optimal threshold (0.421): Precision=59.4%, Recall=72.9%, F1=65.5%
- Analysis: Similar performance to threshold-only but with added complexity

## Key Findings

### 1. Threshold Optimization is Most Effective
- **Highest F1 improvement**: From 63.2% to 68.2% (+5 points)
- **Significant recall gain**: From 53.8% to 73.1% (+19.3 points)
- **Minimal complexity**: No model retraining required
- **Flexible deployment**: Threshold can be tuned for different business scenarios

### 2. Class Weighting Too Aggressive
- Achieves 92.7% recall but precision drops to 45.9%
- Would result in too many false positives for practical marketing use
- Better suited for scenarios where missing positives is extremely costly

### 3. SMOTE Shows Promise but Adds Complexity
- Provides good balance (70.3% recall, 58.2% precision)
- Requires data augmentation and longer training time
- Could be considered for future iterations if threshold optimization proves insufficient

### 4. ROC-AUC Remains Stable
- All approaches maintain strong discriminative performance (0.93+)
- Confirms that the underlying model quality is robust
- Supports threshold-based optimization as low-risk approach

## Selected Approach Rationale

**Chosen**: Threshold Optimization

**Why**:
1. **Best Performance**: Highest F1-score improvement with excellent recall gain
2. **Simplicity**: No model architecture changes required
3. **Flexibility**: Multiple thresholds can be provided for different business needs
4. **Speed**: No additional training time or data augmentation
5. **Interpretability**: Easy to explain to business stakeholders

**Implementation Strategy**: 
- Use precision-recall curve to identify F1-optimal threshold
- Evaluate multiple threshold options (0.5, optimal, 0.3, 0.2)
- Provide comprehensive analysis of business tradeoffs at each threshold

## Future Consideration

If Experiment 2 (threshold optimization) doesn't meet business requirements:
- **Next iteration**: SMOTE oversampling with conservative ratio (0.3-0.4)
- **Alternative**: Ensemble methods combining multiple approaches
- **Advanced**: Cost-sensitive learning with business-specific cost matrix

This systematic exploration provides strong evidence that threshold optimization is the optimal approach for iteration 2, balancing performance gains with implementation simplicity.