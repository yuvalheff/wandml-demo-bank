import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
        if self.config.model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(**self.config.model_params)
        elif self.config.model_type == "random_forest":
            base_model = RandomForestClassifier(**self.config.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Check if hyperparameter tuning is enabled
        if self.config.hyperparameter_tuning.get('enabled', False):
            # Perform hyperparameter tuning
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
                self.model = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_
                self.cv_results_ = grid_search.cv_results_
            else:
                # No parameters to tune, fit directly
                base_model.fit(X, y)
                self.model = base_model
                self.best_params_ = self.config.model_params
        else:
            # Fit without hyperparameter tuning as per experiment plan
            base_model.fit(X, y)
            self.model = base_model
            self.best_params_ = self.config.model_params
        
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