# Experiment 4: Multi-Algorithm Calibrated Ensemble

## Objective
Build an ensemble of calibrated algorithms (Gradient Boosting + XGBoost + LightGBM) to improve upon the strong calibrated gradient boosting performance from Experiment 3 (ROC-AUC: 0.9338).

## Key Change from Previous Iteration
**Single Focus**: Implement multi-algorithm ensemble with soft voting while maintaining the same preprocessing and feature engineering pipeline to isolate ensemble impact.

## Experimental Design

### Data Preprocessing
1. Load train and test datasets from existing CSV files
2. Apply label encoding to categorical columns: `['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16']` using LabelEncoder for each column  
3. Create two engineered binary features:
   - `balance_positive = (V6 > 0).astype(int)`
   - `duration_high = (V12 > V12.quantile(0.75)).astype(int)`
4. Verify final feature set contains 18 features (16 original + 2 engineered)
5. Split features (X) from target column (`target`)

### Feature Engineering
Maintain minimal feature engineering approach from previous successful iterations to isolate ensemble impact:
- **Binary Indicators Only**: Create exactly 2 binary features based on domain logic
- **No Additional Features**: Avoid complex feature engineering to focus purely on ensemble methodology benefits
- **Consistency**: Use same features as Experiment 3 for direct performance comparison

### Model Architecture
**Ensemble Configuration:**
1. **Base Algorithms** (identical hyperparameters):
   - `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)`
   - `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)`  
   - `LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1)`

2. **Calibration Wrapper**: Wrap each algorithm in `CalibratedClassifierCV(cv=3, method='sigmoid')`

3. **Ensemble Method**: `VotingClassifier(voting='soft')` to leverage calibrated probabilities

4. **Training Strategy**: Train ensemble on full training set, evaluate on held-out test set

### Evaluation Strategy

**Primary Target:** ROC-AUC >= 0.935

**Secondary Metrics:**
- Brier Score <= 0.063 (maintain calibration quality)
- Average Precision >= 0.73 (imbalanced data performance)

**Diagnostic Analyses:**
1. **Individual Algorithm Performance** → `individual_algorithms_comparison.html`
   - Compare each base algorithm performance to understand ensemble contribution
   
2. **Ensemble Diversity Analysis** → `ensemble_diversity_analysis.html`  
   - Analyze prediction correlations between algorithms to validate ensemble benefits
   
3. **Calibration Curve Comparison** → `ensemble_calibration_analysis.html`
   - Compare calibration quality of ensemble vs individual algorithms
   
4. **Threshold Optimization** → `ensemble_threshold_optimization.html`
   - Find optimal threshold for ensemble and compare to individual algorithms
   
5. **Business Impact Analysis** → `business_impact_analysis.html`
   - Evaluate precision/recall trade-offs for campaign targeting decisions

### Success Criteria
- ✅ ROC-AUC >= 0.935 on test set (improvement of ~0.0012 over Experiment 3)
- ✅ Brier Score <= 0.063 (maintain calibration quality)
- ✅ F1-Score at optimal threshold >= 0.635 (maintain threshold optimization effectiveness)  
- ✅ Ensemble shows improved diversity compared to single algorithms
- ✅ Model deployable via MLflow with proper ensemble signature

### Expected Deliverables
- `trained_ensemble_model.pkl` - Serialized VotingClassifier ensemble
- `individual_algorithms_comparison.html` - Performance comparison of base algorithms
- `ensemble_diversity_analysis.html` - Correlation analysis between algorithm predictions  
- `ensemble_calibration_analysis.html` - Calibration curves for ensemble vs individual models
- `ensemble_threshold_optimization.html` - Optimal threshold analysis for ensemble
- `business_impact_analysis.html` - Precision/recall analysis for campaign targeting
- `experiment_results.json` - All metrics and performance summaries
- `mlflow_model/` - MLflow logged ensemble model with proper signatures

## Rationale
Previous experiments achieved strong performance with gradient boosting and calibration (ROC-AUC: 0.9338). Exploration experiments demonstrate that multi-algorithm ensemble with calibrated GB, XGBoost, and LightGBM can provide additional performance gains through algorithm diversity while maintaining calibration quality. This approach builds naturally on Experiment 3's calibration success and addresses ensemble recommendations from previous iteration analysis.

## Risk Management
| Risk | Mitigation |
|------|------------|
| Training time increases significantly | Use efficient hyperparameters; acceptable for offline training |
| Ensemble algorithms may be too correlated | Analyze prediction diversity and validate CV performance |
| Model complexity increases deployment | Ensure MLflow compatibility and test inference pipeline |

## Timeline
- **Data Preparation**: ~30 minutes
- **Model Training**: ~45 minutes (3 algorithms + calibration)  
- **Evaluation & Analysis**: ~45 minutes
- **Documentation & MLflow**: ~30 minutes
- **Total**: ~2.5 hours