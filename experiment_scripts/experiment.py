import pandas as pd
import numpy as np
import os
import json
import sklearn
from pathlib import Path
from typing import Dict, List, Any

import mlflow
from mlflow.models.signature import infer_signature

from bank_marketing_campaign_prediction.pipeline.feature_preprocessing import FeatureProcessor
from bank_marketing_campaign_prediction.pipeline.data_preprocessing import DataProcessor
from bank_marketing_campaign_prediction.pipeline.model import ModelWrapper
from bank_marketing_campaign_prediction.config import Config
from bank_marketing_campaign_prediction.model_pipeline import ModelPipeline
from experiment_scripts.evaluation import ModelEvaluator

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)

    def run(self, train_dataset_path: str, test_dataset_path: str, output_dir: str, seed: int = 42) -> Dict[str, Any]:
        """
        Run the complete experiment pipeline.
        
        Parameters:
        train_dataset_path: Path to training data CSV file
        test_dataset_path: Path to test data CSV file  
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility
        
        Returns:
        Dict: Experiment results in required format
        """
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Create output directories
        model_artifacts_dir = os.path.join(output_dir, "model_artifacts")
        general_artifacts_dir = os.path.join(output_dir, "general_artifacts")
        plots_dir = os.path.join(output_dir, "plots")
        
        for dir_path in [model_artifacts_dir, general_artifacts_dir, plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Load datasets
        print("🔄 Loading datasets...")
        train_data = pd.read_csv(train_dataset_path)
        test_data = pd.read_csv(test_dataset_path)
        
        # Separate features and target
        target_col = self._config.data_prep.target_column
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]
        
        print(f"✅ Loaded {len(train_data)} training samples and {len(test_data)} test samples")
        
        # Initialize and fit preprocessing components
        print("🔄 Fitting preprocessing components...")
        
        # Data preprocessing
        data_processor = DataProcessor(self._config.data_prep)
        data_processor.fit(X_train)
        
        # Feature preprocessing  
        feature_processor = FeatureProcessor(
            self._config.feature_prep,
            self._config.data_prep.categorical_columns
        )
        X_train_data_processed = data_processor.transform(X_train)
        feature_processor.fit(X_train_data_processed)
        
        # Apply full preprocessing to training data
        X_train_processed = feature_processor.transform(X_train_data_processed)
        print(f"✅ Processed training data shape: {X_train_processed.shape}")
        
        # Train model
        print("🔄 Training model with hyperparameter optimization...")
        model = ModelWrapper(self._config.model)
        model.fit(X_train_processed, y_train)
        
        print(f"✅ Model training complete. Best params: {model.best_params_}")
        
        # Process test data
        print("🔄 Processing test data...")
        X_test_data_processed = data_processor.transform(X_test)
        X_test_processed = feature_processor.transform(X_test_data_processed)
        print(f"✅ Processed test data shape: {X_test_processed.shape}")
        
        # Evaluate model
        print("🔄 Evaluating model...")
        evaluator = ModelEvaluator(self._config.model_evaluation, output_dir)
        evaluation_results = evaluator.evaluate_model(
            model, X_test_processed, y_test,
            X_train_processed, y_train,
            feature_names=X_train_processed.columns.tolist()
        )
        
        primary_metric_value = evaluation_results[self._config.model_evaluation.primary_metric]
        print(f"✅ Model evaluation complete. {self._config.model_evaluation.primary_metric.upper()}: {primary_metric_value:.4f}")
        
        # Save individual artifacts
        print("🔄 Saving model artifacts...")
        data_processor.save(os.path.join(model_artifacts_dir, "data_processor.pkl"))
        feature_processor.save(os.path.join(model_artifacts_dir, "feature_processor.pkl"))
        model.save(os.path.join(model_artifacts_dir, "trained_model.pkl"))
        
        # Create and test ModelPipeline
        print("🔄 Creating and testing ModelPipeline...")
        pipeline = ModelPipeline(
            data_processor=data_processor,
            feature_processor=feature_processor, 
            model=model
        )
        
        # Test pipeline with sample data
        sample_input = X_test.head(5)
        try:
            sample_predictions = pipeline.predict(sample_input)
            sample_probabilities = pipeline.predict_proba(sample_input)
            print(f"✅ Pipeline test successful. Sample predictions shape: {sample_predictions.shape}")
        except Exception as e:
            raise ValueError(f"Pipeline test failed: {e}")
        
        # Save and log MLflow model
        print("🔄 Saving MLflow model...")
        mlflow_model_dir = os.path.join(model_artifacts_dir, "mlflow_model")
        relative_model_path = "model_artifacts/mlflow_model/"
        
        # Create sample for signature
        signature = infer_signature(sample_input, sample_predictions)
        
        # 1. Always save model locally
        print(f"💾 Saving model to local disk: {mlflow_model_dir}")
        mlflow.sklearn.save_model(
            pipeline,
            path=mlflow_model_dir,
            code_paths=["bank_marketing_campaign_prediction"],
            signature=signature
        )
        
        # 2. Log to active MLflow run if available
        active_run_id = "c74be0beb1ea43b5bbb31bc9f60e78c4"
        logged_model_uri = None
        
        if active_run_id and active_run_id != 'None' and active_run_id.strip():
            print(f"✅ Logging model to MLflow run: {active_run_id}")
            try:
                with mlflow.start_run(run_id=active_run_id):
                    logged_model_info = mlflow.sklearn.log_model(
                        pipeline,
                        artifact_path="model",
                        code_paths=["bank_marketing_campaign_prediction"],
                        signature=signature
                    )
                    logged_model_uri = logged_model_info.model_uri
                    print(f"✅ Model logged to: {logged_model_uri}")
            except Exception as e:
                print(f"⚠️  Warning: Could not log to MLflow run: {e}")
                logged_model_uri = None
        else:
            print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
        
        # Save evaluation results to general artifacts
        with open(os.path.join(general_artifacts_dir, "evaluation_results.json"), 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_to_json_serializable(evaluation_results)
            json.dump(json_results, f, indent=2)
        
        # Create input example for MLflow model info
        input_example = sample_input.head(1).to_dict('records')[0]
        
        # Prepare return results
        results = {
            "metric_name": self._config.model_evaluation.primary_metric,
            "metric_value": float(primary_metric_value),
            "model_artifacts": [
                "data_processor.pkl",
                "feature_processor.pkl", 
                "trained_model.pkl",
                "mlflow_model/"
            ],
            "mlflow_model_info": {
                "model_path": "model_artifacts/mlflow_model/",
                "logged_model_uri": logged_model_uri,
                "model_type": "sklearn",
                "task_type": "classification",
                "signature": signature.to_dict() if signature else None,
                "input_example": input_example,
                "framework_version": sklearn.__version__
            }
        }
        
        print(f"🎉 Experiment completed successfully!")
        print(f"📊 Primary metric ({self._config.model_evaluation.primary_metric}): {primary_metric_value:.4f}")
        print(f"📁 Artifacts saved to: {output_dir}")
        
        return results
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj