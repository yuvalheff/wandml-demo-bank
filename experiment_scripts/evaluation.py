from bank_marketing_campaign_prediction.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
