import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

from bank_marketing_campaign_prediction.config import ModelConfig


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = None
        self.best_params_ = None
        self.cv_results_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        # Initialize the base model with exact hyperparameters as specified in experiment plan
        # Filter out calibration parameters that don't belong to the base model
        base_model_params = {k: v for k, v in self.config.model_params.items() 
                           if k not in ['calibration_enabled', 'calibration_method', 'calibration_cv']}
        
        if self.config.model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(**base_model_params)
        elif self.config.model_type == "random_forest":
            base_model = RandomForestClassifier(**base_model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Check if hyperparameter tuning is enabled
        if self.config.hyperparameter_tuning.get('enabled', False):
            # Perform hyperparameter tuning on the base model first
            param_grid = {}
            for key, value in self.config.hyperparameter_tuning.items():
                if key not in ['enabled', 'cv_folds'] and isinstance(value, list):
                    param_grid[key] = value
            
            if param_grid:  # Only do grid search if there are parameters to search
                cv = StratifiedKFold(
                    n_splits=self.config.hyperparameter_tuning.get('cv_folds', 5), 
                    shuffle=True, 
                    random_state=self.config.model_params['random_state']
                )
                
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X, y)
                base_model = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_
                self.cv_results_ = grid_search.cv_results_
            else:
                # No parameters to tune, fit directly
                base_model.fit(X, y)
                self.best_params_ = self.config.model_params
        else:
            # Fit without hyperparameter tuning as per experiment plan
            # Don't fit the base model yet, let CalibratedClassifierCV do it
            self.best_params_ = self.config.model_params
        
        # Apply calibration as specified in experiment plan
        # Use CalibratedClassifierCV with sigmoid calibration method and 3-fold CV
        calibration_enabled = self.config.model_params.get('calibration_enabled', False)
        if calibration_enabled:
            calibration_method = self.config.model_params.get('calibration_method', 'sigmoid')
            calibration_cv = self.config.model_params.get('calibration_cv', 3)
            
            self.model = CalibratedClassifierCV(
                estimator=base_model,
                method=calibration_method,
                cv=calibration_cv
            )
            self.model.fit(X, y)
        else:
            # No calibration - fit base model directly
            if not hasattr(base_model, 'n_estimators') or base_model.n_estimators is None:
                # Model wasn't fitted yet during hyperparameter tuning
                base_model.fit(X, y)
            self.model = base_model
        
        return self

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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from the fitted model.
        
        Returns:
        Dict[str, float]: Feature names and their importance scores.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None
    
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