from typing import Optional, List
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

from bank_marketing_campaign_prediction.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config
        self.feature_columns = None  # Will store final feature columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # Store the final feature columns (all 16 original + 2 engineered)
        # DataProcessor should have already created V14_is_missing and V15_is_zero
        self.feature_columns = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        # Since DataProcessor handles all preprocessing and feature engineering,
        # FeatureProcessor just ensures consistency and validates features
        X_processed = X.copy()
        
        # Ensure all expected binary features are present
        expected_binary_features = self.config.binary_features
        for feature in expected_binary_features:
            if feature not in X_processed.columns:
                # Create missing binary feature with default value 0
                X_processed[feature] = 0
                
        # Ensure columns match training data if fitted
        if self.feature_columns is not None:
            # Add missing columns with zeros
            for col in self.feature_columns:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            
            # Select only the columns that were present during training
            X_processed = X_processed[self.feature_columns]
        
        return X_processed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input features.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the feature processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FeatureProcessor':
        """
        Load the feature processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        FeatureProcessor: The loaded feature processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
