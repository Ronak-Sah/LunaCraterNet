from src.config.configuration import ConfigurationManager
from src.components.library.Resize import Cropper
from pathlib import Path
import os
from src.components.library.vectors import VectorDb
from src.logger import logger
folder_path=Path("D:\\Ml Dl\\Project\\LunaCraterNet\\artifacts\\data_ingestion\\dataset\\LU3M6TGT_yolo_format\\valid\\images")

class vector_db_pipeline:
    def __init__(self):
        pass
    
    def main(self,total=20):
        config=ConfigurationManager()
        lib_config=config.get_library_components()
        all_images=os.listdir(folder_path)[:total]
        
        for image in  all_images:
            image_path=os.path.join(folder_path,image)
            im=Cropper(image_path,lib_config)
            im.crop()
        logger.info(f"Cropped images saves at {lib_config.crop_save_dir}") 
        logger.info("Cropping done")
        vec=VectorDb(lib_config)
        vec.storeDb()
        
        