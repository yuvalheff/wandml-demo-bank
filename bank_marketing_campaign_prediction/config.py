from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


class ConfigParsingFailed(Exception):
    pass


@dataclass
class DataConfig:
    version: str
    dataset_name: str
    target_column: str
    categorical_columns: List[str]
    numerical_columns: List[str]
    v14_missing_threshold: float
    v6_outlier_bounds: Dict[str, float]


@dataclass
class FeaturesConfig:
    encoding_method: str
    drop_first: bool
    handle_unknown: str
    binary_features: List[str]


@dataclass
class ModelEvalConfig:
    cv_folds: int
    primary_metric: str
    random_state: int
    threshold_optimization: bool
    threshold_metric: str
    thresholds_to_evaluate: List[float]


@dataclass
class ModelConfig:
    model_type: str
    model_params: Dict[str, Any]
    hyperparameter_tuning: Dict[str, Any]
    baseline_model: Dict[str, Any]


@dataclass
class Config:
    data_prep: DataConfig
    feature_prep: FeaturesConfig
    model_evaluation: ModelEvalConfig
    model: ModelConfig

    @staticmethod
    def from_yaml(config_file: str):
        with open(config_file, 'r', encoding='utf-8') as stream:
            try:
                config_data = yaml.safe_load(stream)
                return Config(
                    data_prep=DataConfig(**config_data['data_prep']),
                    feature_prep=FeaturesConfig(**config_data['feature_prep']),
                    model_evaluation=ModelEvalConfig(**config_data['model_evaluation']),
                    model=ModelConfig(**config_data['model'])
                )
            except (yaml.YAMLError, OSError) as e:
                raise ConfigParsingFailed from e