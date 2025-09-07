from typing import Optional
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from bank_marketing_campaign_prediction.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config
        self.label_encoders = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        DataProcessor: The fitted processor.
        """
        X_copy = X.copy()
        
        # Fit label encoders for categorical columns
        for col in self.config.categorical_columns:
            if col in X_copy.columns:
                # Convert to string first as specified in the experiment plan
                X_copy[col] = X_copy[col].astype(str)
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(X_copy[col])
        
        # Fit duration threshold for duration_high feature (75th percentile)
        if 'V12' in X_copy.columns:
            self.duration_threshold_ = X_copy['V12'].quantile(0.75)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        X_transformed = X.copy()
        
        # 1. Apply LabelEncoder to categorical columns (convert to string first)
        for col in self.config.categorical_columns:
            if col in X_transformed.columns and col in self.label_encoders:
                X_transformed[col] = X_transformed[col].astype(str)
                # Handle unseen categories by assigning them to the first class
                try:
                    X_transformed[col] = self.label_encoders[col].transform(X_transformed[col])
                except ValueError:
                    # Handle unseen values by mapping them to the first class (0)
                    mask = ~X_transformed[col].isin(self.label_encoders[col].classes_)
                    X_transformed.loc[mask, col] = self.label_encoders[col].classes_[0]
                    X_transformed[col] = self.label_encoders[col].transform(X_transformed[col])
        
        # 2. Create binary features: balance_positive for V6 > 0
        if 'V6' in X_transformed.columns:
            X_transformed['balance_positive'] = (X_transformed['V6'] > 0).astype(int)
        
        # 3. Create binary features: duration_high for V12 > 75th percentile
        if 'V12' in X_transformed.columns:
            # Note: This requires fitting on training data to get the quantile
            if not hasattr(self, 'duration_threshold_'):
                # If not fitted, calculate on current data (for transform only)
                self.duration_threshold_ = X_transformed['V12'].quantile(0.75)
            X_transformed['duration_high'] = (X_transformed['V12'] > self.duration_threshold_).astype(int)
        
        # 4. Apply np.clip to V6 column with bounds (-1000, 10000) to handle extreme outliers
        if 'V6' in X_transformed.columns:
            X_transformed['V6'] = np.clip(
                X_transformed['V6'], 
                self.config.v6_outlier_bounds['lower'], 
                self.config.v6_outlier_bounds['upper']
            )
        
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the data processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'DataProcessor':
        """
        Load the data processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        DataProcessor: The loaded data processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
