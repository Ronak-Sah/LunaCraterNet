from ultralytics import YOLO
from src.logger import logger
from src.config.configuration import ConfigurationManager
import os
from pathlib import Path


class Model:
    def __init__(self):
        config=ConfigurationManager()
        eval_config=config.get_model_evaluation()
        self.config= eval_config
     
        self.model=YOLO(self.config.model_path)

    def predict(self,path:Path):
        results=self.model(path)

        for result in results:
            result.show()