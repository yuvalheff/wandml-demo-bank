from typing import Optional
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

from bank_marketing_campaign_prediction.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        DataProcessor: The fitted processor.
        """
        # Calculate outlier thresholds for V6
        if 'V6' in X.columns:
            self.v6_q01 = X['V6'].quantile(self.config.v6_outlier_percentiles['lower'])
            self.v6_q99 = X['V6'].quantile(self.config.v6_outlier_percentiles['upper'])
        else:
            self.v6_q01 = None
            self.v6_q99 = None
        
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
        
        # 1. Handle special values for V14 - Create binary indicator for V14==-1.0
        if 'V14' in X_transformed.columns:
            X_transformed['V14_is_missing'] = (X_transformed['V14'] == self.config.v14_missing_threshold).astype(int)
        
        # 2. Handle special values for V15 - Create binary indicator for V15==0.0
        if 'V15' in X_transformed.columns:
            X_transformed['V15_is_zero'] = (X_transformed['V15'] == 0.0).astype(int)
        
        # 3. Outlier treatment for V6 - Cap using 99th and 1st percentile
        if 'V6' in X_transformed.columns and self.v6_q01 is not None and self.v6_q99 is not None:
            X_transformed['V6'] = np.clip(X_transformed['V6'], self.v6_q01, self.v6_q99)
        
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
