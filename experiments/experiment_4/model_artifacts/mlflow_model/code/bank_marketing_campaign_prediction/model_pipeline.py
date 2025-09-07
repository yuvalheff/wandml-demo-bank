"""
ModelPipeline for Bank Marketing Campaign Prediction

Complete ML pipeline that combines data preprocessing, feature engineering, 
and model prediction for MLflow deployment.
"""

import pandas as pd
import numpy as np
from typing import Any

from bank_marketing_campaign_prediction.pipeline.data_preprocessing import DataProcessor
from bank_marketing_campaign_prediction.pipeline.feature_preprocessing import FeatureProcessor
from bank_marketing_campaign_prediction.pipeline.model import ModelWrapper


class ModelPipeline:
    """
    Complete ML pipeline for bank marketing campaign prediction.
    
    Combines data preprocessing, feature engineering, and model prediction
    into a single deployable unit suitable for MLflow.
    """
    
    def __init__(self, data_processor: DataProcessor = None, 
                 feature_processor: FeatureProcessor = None,
                 model: ModelWrapper = None):
        """
        Initialize the pipeline components.
        
        Parameters:
        data_processor: Fitted DataProcessor instance
        feature_processor: Fitted FeatureProcessor instance  
        model: Fitted ModelWrapper instance
        """
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        self.model = model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on raw input data.
        
        Parameters:
        X (pd.DataFrame): Raw input features
        
        Returns:
        np.ndarray: Predicted class labels
        """
        # Validate that all components are fitted
        if not all([self.data_processor, self.feature_processor, self.model]):
            raise ValueError("Pipeline components not properly initialized. Ensure all components are fitted.")
        
        # Apply preprocessing pipeline
        X_processed = self._preprocess(X)
        
        # Make predictions
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities on raw input data.
        
        Parameters:
        X (pd.DataFrame): Raw input features
        
        Returns:
        np.ndarray: Predicted class probabilities
        """
        # Validate that all components are fitted
        if not all([self.data_processor, self.feature_processor, self.model]):
            raise ValueError("Pipeline components not properly initialized. Ensure all components are fitted.")
        
        # Apply preprocessing pipeline
        X_processed = self._preprocess(X)
        
        # Make probability predictions
        return self.model.predict_proba(X_processed)
    
    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the complete preprocessing pipeline to raw data.
        
        Parameters:
        X (pd.DataFrame): Raw input features
        
        Returns:
        pd.DataFrame: Fully processed features ready for model
        """
        # Step 1: Data preprocessing (outlier handling, special value indicators)
        X_data_processed = self.data_processor.transform(X)
        
        # Step 2: Feature preprocessing (one-hot encoding, column alignment)
        X_feature_processed = self.feature_processor.transform(X_data_processed)
        
        return X_feature_processed
