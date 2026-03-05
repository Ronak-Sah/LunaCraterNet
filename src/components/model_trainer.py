import os
import torch
from src.logger import logger
from src.entity import ModelTrainerConfig
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class Model_Trainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config= config   
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        
        model_path=Path(self.config.model_path)
        if model_path.exists():
            logger.info("Resuming training...")
            self.model=YOLO(model_path)
        else:
            self.model = YOLO("yolov8s.pt").to(self.device)
              

    def train(self):

            results=self.model.train(
                project=os.path.join(os.getcwd(), "artifacts", "model_trainer"),
                name="runs",
                data=self.config.data_path,
                epochs=self.config.epochs,
                batch=self.config.batch_size,
                imgsz=416,
                device=self.device,
                fraction=0.5,
                exist_ok=True,
                val=False
            )

            self.model.export(format="onnx", 
                device=0,
                project=self.config.root_dir,
                name="yolo"
            )
            
            logger.info("Training Complete.")

    