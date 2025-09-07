from typing import Optional, List
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

from bank_marketing_campaign_prediction.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig, categorical_columns: List[str]):
        self.config: FeaturesConfig = config
        self.categorical_columns = categorical_columns
        self.feature_columns = None  # Will store final feature columns after encoding

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # Apply one-hot encoding to get final feature columns
        X_encoded = self._apply_encoding(X)
        self.feature_columns = X_encoded.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        # Apply one-hot encoding
        X_encoded = self._apply_encoding(X)
        
        # Ensure columns match training data
        if self.feature_columns is not None:
            # Add missing columns with zeros
            for col in self.feature_columns:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0
            
            # Select only the columns that were present during training
            X_encoded = X_encoded[self.feature_columns]
        
        return X_encoded
    
    def _apply_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns.
        """
        if self.config.encoding_method == "one_hot":
            # Identify categorical columns that exist in the data
            existing_cat_cols = [col for col in self.categorical_columns if col in X.columns]
            
            if existing_cat_cols:
                X_encoded = pd.get_dummies(
                    X, 
                    columns=existing_cat_cols,
                    drop_first=self.config.drop_first,
                    dtype=int  # Ensure integer type for compatibility
                )
            else:
                X_encoded = X.copy()
        else:
            X_encoded = X.copy()
        
        return X_encoded

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
