import os
import torch
from src.logger import logger
from src.entity import ModelTrainerConfig
from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2 
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss



class Model_Trainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config= config
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        folder_path=os.path.join(os.getcwd(), "artifacts", "model_trainer")
        if folder_path.exists():
            logger.info("Resuming training...")
            model_path="artifacts\\model_trainer\\runs\\weights\\best.pt"
            self.model=YOLO(model_path)
        else:
            self.model = YOLO("yolov8s.pt").to(self.device)
              

    def train(self):

            results=self.model.train(
                project=os.path.join(os.getcwd(), "artifacts", "model_trainer"),
                name="runs",
                data='artifacts\data_ingestion\dataset\LU3M6TGT_yolo_format\data.yaml',
                epochs=self.config.epochs,
                batch=self.config.batch_size,
                imgsz=416,
                device=self.device,
                fraction=0.5,
                exist_ok=True,
                val=False
            )

            # self.model.export(format="engine", 
            #     device=0,
            #     project="artifacts/modeltrainer",
            #     name="yolo"
            # )
            
            logger.info("Training Complete.")

    