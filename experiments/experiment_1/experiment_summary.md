# Experiment 1: Gradient Boosting Baseline with Optimized Preprocessing

## Experiment Overview
**Task**: Binary classification for bank marketing campaign prediction  
**Target**: Term deposit subscription prediction (yes/no)  
**Primary Metric**: ROC-AUC  
**Date**: 2025-09-07  

## Results Summary
✅ **Target Achieved**: ROC-AUC of **0.935** met the expected target of ≥0.935  
🎯 **Strong Performance**: Model demonstrates excellent discriminative ability with high ROC-AUC  
📊 **Class Imbalance**: Model handles imbalanced dataset well (529 positive vs 3993 negative samples)

## Key Metrics
- **ROC-AUC**: 0.935 (Target: ≥0.935) ✅
- **Average Precision**: 0.733
- **Brier Score**: 0.062 (lower is better)
- **Accuracy**: 90.98%
- **Precision (Class 2)**: 66.05%
- **Recall (Class 2)**: 47.07%
- **F1-Score (Class 2)**: 54.97%

## Model Configuration
- **Algorithm**: Gradient Boosting Classifier
- **Best Hyperparameters**:
  - learning_rate: 0.1
  - max_depth: 5
  - min_samples_split: 10  
  - n_estimators: 200
- **Cross-Validation**: 5-fold stratified CV
- **Features**: 44 processed features after one-hot encoding

## Preprocessing Steps Applied
1. **Special Value Handling**: Created binary indicators for V14==-1.0 and V15==0.0
2. **Outlier Treatment**: Applied 99th/1st percentile clipping to V6 feature
3. **Categorical Encoding**: One-hot encoding with drop_first=True for 9 categorical features
4. **Column Alignment**: Ensured training/test feature consistency

## Key Strengths
1. **Met Performance Target**: Successfully achieved ROC-AUC ≥ 0.935
2. **Robust Hyperparameter Optimization**: Found optimal configuration through systematic grid search
3. **Comprehensive Evaluation**: Generated 7 diagnostic plots for thorough analysis
4. **Pipeline Reliability**: Created complete MLflow model pipeline for deployment

## Main Weaknesses
1. **Class Imbalance Issues**: Recall for positive class (47.07%) indicates difficulty capturing all positive cases
2. **Precision-Recall Trade-off**: While ROC-AUC is excellent, precision-recall performance shows room for improvement
3. **Limited Feature Engineering**: Basic preprocessing may miss interaction effects or advanced feature relationships

## Experiment vs Plan Analysis
**Planning Hypothesis**: Gradient Boosting with optimized preprocessing would achieve ROC-AUC ≥ 0.935
- ✅ **Hypothesis Confirmed**: Achieved exactly 0.935 ROC-AUC
- ✅ **Preprocessing Effective**: V12 importance validated, categorical encoding successful
- ✅ **Model Selection Justified**: Gradient Boosting performed as expected

## Future Improvement Suggestions
1. **Address Class Imbalance**: Implement SMOTE, class weighting, or threshold optimization to improve recall
2. **Advanced Feature Engineering**: Create interaction features, polynomial terms, or domain-specific features
3. **Ensemble Methods**: Explore stacking or blending multiple algorithms for enhanced performance
4. **Threshold Optimization**: Find optimal prediction threshold for business requirements

## Context for Next Iteration
- Strong baseline established with ROC-AUC of 0.935
- Model demonstrates good generalization capability
- Main opportunity lies in improving minority class detection
- Feature engineering has potential for further gains
- Current preprocessing pipeline provides solid foundation for advanced techniques

## Artifacts Generated
- 4 model artifacts: data_processor.pkl, feature_processor.pkl, trained_model.pkl, mlflow_model/
- 7 evaluation plots: ROC curve, precision-recall, confusion matrix, calibration, feature importance, prediction distribution, threshold analysis
- Complete MLflow model ready for deployment

## Business Impact
The model successfully identifies potential term deposit subscribers with 93.5% discriminative accuracy. The current configuration prioritizes overall performance but may miss approximately 53% of potential subscribers (recall=47%). For marketing campaigns, this suggests the model reliably identifies high-probability targets but may benefit from threshold tuning based on campaign budget and response rate requirements.