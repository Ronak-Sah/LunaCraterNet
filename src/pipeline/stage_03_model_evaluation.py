from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import Model_Evaluation

class Model_Evaluation_pipeline:
    def __init__(self):
        pass
    
    def main(self):
        config=ConfigurationManager()
        eval_config=config.get_model_evaluation()

        model_eval=Model_Evaluation(eval_config)
        model_eval.evaluate()
        
        