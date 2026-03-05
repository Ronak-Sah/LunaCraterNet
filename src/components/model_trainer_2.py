from pathlib import Path
import cv2
from src.components.inference import Model
import os
import torch

class Cropper:
    def __init__(self,path:Path):
        model=Model()
        self.image_path=path
        self.results=model.predict(path)
        self.save_dir=Path("D://Ml Dl//Project//LunaCraterNet//artifacts//crops")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def crop(self):
        image=cv2.imread(self.image_path)

        boxes = self.results[0].boxes.xyxy  
        conf = self.results[0].boxes.conf
        h, w = image.shape[:2]
        for i, box in enumerate(boxes):
            if conf[i]>0.7:
                x1, y1, x2, y2 = map(int, box)

                bw = x2 - x1
                bh = y2 - y1

                pad_x = int(bw * 0.5)
                pad_y = int(bh * 0.5)

                x1_new = max(0, x1 - pad_x)
                y1_new = max(0, y1 - pad_y)
                x2_new = min(w, x2 + pad_x)
                y2_new = min(h, y2 + pad_y)

                crop = image[y1_new:y2_new, x1_new:x2_new]
                path=f"crop_{i}.jpg"
                save_path = os.path.join(self.save_dir,path)
                cv2.imwrite(save_path, crop)
                
        print("Cropping done")
        return crop


