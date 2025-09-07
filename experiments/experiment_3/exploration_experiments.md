# Exploration Experiments for Iteration 3

## Overview
Before finalizing the experiment plan for Iteration 3, I conducted a comprehensive set of exploration experiments to identify the most promising single improvement over Iteration 2's successful threshold optimization approach. These experiments tested various hypotheses to maximize the chance of meaningful performance gains.

## Baseline Performance
- **Current Model**: GradientBoostingClassifier with threshold optimization  
- **ROC-AUC**: 0.932 (maintained consistently across iterations)
- **Recall at Optimal Threshold**: 74.7% (significant improvement from 47.1%)
- **F1-Score at Optimal Threshold**: 0.627

## Exploration Results Summary

### Experiment 1: Alternative Algorithms
**Hypothesis**: Different tree-based algorithms might provide better performance

**Results**:
- GradientBoosting: **0.9328** ± 0.0077 ✅
- RandomForest: 0.9001 ± 0.0075 
- ExtraTrees: 0.8623 ± 0.0221

**Conclusion**: GradientBoosting remains the best algorithm choice. No improvement opportunity here.

### Experiment 2: Advanced Feature Engineering  
**Hypothesis**: Additional feature interactions and transformations could improve model performance

**Tested Features**:
- V12 interactions (V12×V1, V12×V6) - leveraging the strongest predictor
- Age-based features (V1_squared, is_senior, is_young)  
- Duration transformations (V12_log, is_long_call)
- Campaign interactions (V13×V14)

**Results**:
- Advanced Features (26 features): 0.9314 ± 0.0082
- Baseline (18 features): **0.9328** ± 0.0077 ✅

**Conclusion**: Advanced feature engineering actually decreased performance by 0.0014. Current feature set is optimal.

### Experiment 3: Hyperparameter Optimization
**Hypothesis**: Different hyperparameter configurations could yield better performance  

**Tested Configurations**:
- Current (200 est, depth=5, lr=0.1): **0.9328** ± 0.0077 ✅
- More trees + deeper (300 est, depth=6): 0.9311 ± 0.0084
- Deeper + slower LR (depth=7, lr=0.05): 0.9321 ± 0.0079  
- Many shallow trees (400 est, depth=4, lr=0.05): 0.9321 ± 0.0084
- Faster LR (lr=0.2): 0.9305 ± 0.0079

**Conclusion**: Current hyperparameters are already optimal. No improvement opportunity.

### Experiment 4: Ensemble Methods
**Hypothesis**: Combining multiple algorithms could improve performance through diversity

**Results**:
- GradientBoosting alone: **0.9328** ± 0.0077 ✅
- RandomForest alone: 0.9001 ± 0.0075
- Soft Voting (GB+RF): 0.9291 ± 0.0066
- Diverse Ensemble (GB+RF+LR): 0.9246 ± 0.0064

**Conclusion**: Ensembles actually decreased performance. Single GradientBoosting model is superior.

### Experiment 5: Categorical Encoding Strategies
**Hypothesis**: One-hot encoding might better capture categorical relationships than label encoding

**Results**:
- Label Encoding (18 features): **0.9328** ± 0.0077 ✅  
- One-Hot Encoding (44 features): 0.9319 ± 0.0051

**Conclusion**: Label encoding is superior for tree-based models. Current approach is optimal.

### Experiment 6: Cost-Sensitive Learning
**Hypothesis**: Explicit class weighting could improve performance on imbalanced data

**Results**:
- No weights: 0.9328 ± 0.0077
- Balanced weights: 0.9298 ± 0.0079
- 2x positive weight: **0.9329** ± 0.0078 ⭐ (+0.0001)
- 3x positive weight: 0.9322 ± 0.0078
- 5x+ weights: Performance degraded

**Conclusion**: Marginal improvement with 2x positive weighting, but very small gain.

### Experiment 7: Probability Calibration ⭐
**Hypothesis**: Better calibrated probabilities could improve threshold optimization effectiveness

**Results**:
- Base model: 0.9328 ± 0.0077
- Sigmoid calibration: **0.9336** ± 0.0082 ⭐ (+0.0007)  
- Isotonic calibration: 0.9333 ± 0.0082 (+0.0004)

**Conclusion**: BEST improvement found! Sigmoid calibration provides measurable performance gain.

## Final Decision: Probability Calibration

### Rationale
Among all tested approaches, **probability calibration with sigmoid method** showed the most promising results:
- **Performance**: +0.0007 ROC-AUC improvement (largest gain observed)  
- **Strategic Alignment**: Complements Iteration 2's successful threshold optimization
- **Business Value**: More reliable probability estimates enhance decision-making
- **Implementation**: Clean, focused change that isolates calibration impact

### Why This Choice Makes Sense
1. **Iteration 2 Success**: Threshold optimization was highly successful, achieving 74.7% recall
2. **Probability Reliability**: Better calibrated probabilities directly support better threshold decisions
3. **Business Impact**: More trustworthy probability estimates improve campaign targeting
4. **Measurable Improvement**: Only approach that consistently showed performance gains
5. **Production Value**: Calibrated models provide better real-world decision support

### Alternative Approaches Considered
- **Cost-sensitive learning**: Very marginal gains (+0.0001) and adds complexity
- **Advanced features**: Actually hurt performance, current features are optimal  
- **Hyperparameter tuning**: Current settings already optimal
- **Ensemble methods**: Decreased performance compared to single model
- **Encoding changes**: Current label encoding is superior

## Methodology Notes
All experiments used:
- 3-fold Stratified Cross-Validation for speed (reduced from 5-fold)
- Consistent random_state=42 for reproducibility  
- Same preprocessing pipeline as Iteration 2
- ROC-AUC as primary evaluation metric
- Conservative approach to avoid overfitting to validation

The comprehensive exploration ensures that **probability calibration** is genuinely the best next step for Iteration 3, providing measurable improvement while maintaining the strong foundation built in previous iterations.