from src.logger import logger 
from multiprocessing import freeze_support

from src.pipeline.stage_01_data_ingestion import Data_Ingestion_pipeline

def main():
    logger.info("Code Starts")
    data_ingestion=Data_Ingestion_pipeline()
    data_ingestion.main()
    
    

if __name__ == "__main__":
    freeze_support()   
    main()
