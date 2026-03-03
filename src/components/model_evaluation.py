from ultralytics import YOLO
import torch
from src.logger import logger
from src.entity import ModelEvalConfig
import os

class Model_Evaluation:
    def __init__(self,config: ModelEvalConfig):
        self.config= config
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
     
        self.model=YOLO(self.config.model_path)

    def evaluate(self):
        metrics=self.model.val(
            project=os.path.join(os.getcwd(), "artifacts", "model_evaluation"),
            name="runs",
            data=self.config.data_path,
            batch=self.config.batch_size,
            imgsz=416,
            device=self.device,
            fraction=0.1,
            exist_ok=True
        )
        logger.info("Evaluation Completed")