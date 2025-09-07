import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List

from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

from bank_marketing_campaign_prediction.config import ModelConfig


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = None
        self.base_estimators = {}
        self.best_params_ = None
        self.cv_results_ = None

    def _create_base_estimator(self, algorithm: str, params: Dict[str, Any]):
        """Create a base estimator based on algorithm name and parameters."""
        if algorithm == "gradient_boosting":
            return GradientBoostingClassifier(**params)
        elif algorithm == "xgboost":
            return xgb.XGBClassifier(**params, eval_metric='logloss')
        elif algorithm == "lightgbm":
            return lgb.LGBMClassifier(**params, verbose=-1)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the multi-algorithm ensemble classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        if self.config.model_type == "multi_algorithm_ensemble":
            self._fit_ensemble(X, y)
        else:
            self._fit_single_model(X, y)
        
        return self

    def _fit_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Fit the multi-algorithm ensemble model."""
        base_algorithms = self.config.model_params.get('base_algorithms', [])
        voting_type = self.config.model_params.get('voting_type', 'soft')
        calibration_enabled = self.config.model_params.get('calibration_enabled', False)
        calibration_method = self.config.model_params.get('calibration_method', 'sigmoid')
        calibration_cv = self.config.model_params.get('calibration_cv', 3)
        
        # Create base estimators
        estimators = []
        for i, algo_config in enumerate(base_algorithms):
            algorithm = algo_config['algorithm']
            params = algo_config['params']
            
            # Create base estimator
            base_estimator = self._create_base_estimator(algorithm, params)
            
            # Apply calibration if enabled
            if calibration_enabled:
                calibrated_estimator = CalibratedClassifierCV(
                    estimator=base_estimator,
                    method=calibration_method,
                    cv=calibration_cv
                )
                estimator_name = f"calibrated_{algorithm}_{i}"
                estimators.append((estimator_name, calibrated_estimator))
                self.base_estimators[estimator_name] = calibrated_estimator
            else:
                estimator_name = f"{algorithm}_{i}"
                estimators.append((estimator_name, base_estimator))
                self.base_estimators[estimator_name] = base_estimator
        
        # Create voting ensemble
        self.model = VotingClassifier(
            estimators=estimators,
            voting=voting_type
        )
        
        # Fit the ensemble
        self.model.fit(X, y)
        self.best_params_ = self.config.model_params

    def _fit_single_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit a single model (fallback for backwards compatibility)."""
        # Filter out ensemble-specific parameters
        model_params = {k: v for k, v in self.config.model_params.items() 
                       if k not in ['base_algorithms', 'ensemble_method', 'voting_type']}
        
        base_model_params = {k: v for k, v in model_params.items() 
                           if k not in ['calibration_enabled', 'calibration_method', 'calibration_cv']}
        
        if self.config.model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(**base_model_params)
        elif self.config.model_type == "xgboost":
            base_model = xgb.XGBClassifier(**base_model_params, eval_metric='logloss')
        elif self.config.model_type == "lightgbm":
            base_model = lgb.LGBMClassifier(**base_model_params, verbose=-1)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Apply calibration if enabled
        calibration_enabled = model_params.get('calibration_enabled', False)
        if calibration_enabled:
            calibration_method = model_params.get('calibration_method', 'sigmoid')
            calibration_cv = model_params.get('calibration_cv', 3)
            
            self.model = CalibratedClassifierCV(
                estimator=base_model,
                method=calibration_method,
                cv=calibration_cv
            )
        else:
            self.model = base_model
        
        self.model.fit(X, y)
        self.best_params_ = model_params

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importances from the fitted model.
        For ensemble models, returns importance from each base estimator.
        
        Returns:
        Dict[str, Any]: Feature importance information.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if isinstance(self.model, VotingClassifier):
            # For ensemble, get feature importance from each base estimator
            importances = {}
            for name, estimator in self.model.named_estimators_.items():
                # Handle calibrated classifiers
                if isinstance(estimator, CalibratedClassifierCV):
                    # Get importance from the base estimator of the first calibrated classifier
                    if hasattr(estimator.calibrated_classifiers_[0].estimator, 'feature_importances_'):
                        importances[name] = estimator.calibrated_classifiers_[0].estimator.feature_importances_
                elif hasattr(estimator, 'feature_importances_'):
                    importances[name] = estimator.feature_importances_
            return importances
        elif hasattr(self.model, 'feature_importances_'):
            return {'model': self.model.feature_importances_}
        else:
            return None

    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble composition.
        
        Returns:
        Dict[str, Any]: Ensemble information including estimator names and types.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        info = {
            'model_type': self.config.model_type,
            'is_ensemble': isinstance(self.model, VotingClassifier),
            'base_estimators': {}
        }
        
        if isinstance(self.model, VotingClassifier):
            info['voting_type'] = self.model.voting
            info['n_estimators'] = len(self.model.estimators_)
            
            for name, estimator in self.model.named_estimators_.items():
                est_info = {
                    'type': type(estimator).__name__,
                    'is_calibrated': isinstance(estimator, CalibratedClassifierCV)
                }
                if isinstance(estimator, CalibratedClassifierCV):
                    est_info['base_estimator_type'] = type(estimator.estimator).__name__
                    est_info['calibration_method'] = estimator.method
                    est_info['cv_folds'] = estimator.cv
                
                info['base_estimators'][name] = est_info
        
        return info
    
    def save(self, path: str) -> None:
        """
        Save the model wrapper as an artifact.
        
        Parameters:
        path (str): The file path to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'ModelWrapper':
        """
        Load the model wrapper from a saved artifact.
        
        Parameters:
        path (str): The file path to load the model from.
        
        Returns:
        ModelWrapper: The loaded model wrapper.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)