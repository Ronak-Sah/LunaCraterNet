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
        results=self.model(path, conf=0.7, iou=0.6,max_det=50)

        # for result in results:
        #     boxes = result.boxes
        #     for box in boxes.xyxy:
        #         x1, y1, x2, y2 = box.tolist()
        #         print(x1, y1, x2, y2)
        
        #     result.show()

        return results